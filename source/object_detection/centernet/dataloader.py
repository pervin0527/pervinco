import os
import cv2
import math
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from six import raise_from


def findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, txt_filename, classes, batch_size, input_shape, max_detections):
        self.data_dir = data_dir
        self.lines = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, txt_filename + '.txt')).readlines()]
        self.classes = classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_size = input_shape[0] // 4
        self.max_detections = max_detections

        self.current_index = 0
        self.groups = None
        self.group_images()


    def __len__(self):
        return len(self.lines)


    def group_images(self):
        order = list(range(len(self.lines)))
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


    def __getitem__(self, index):
        group = self.groups[self.current_index]
        inputs, targets = self.compute_inputs_targets(group)

        while inputs is None:
            current_index = self.current_index + 1

            if current_index >= len(self.groups):
                current_index = current_index % (len(self.groups))

            self.current_index = current_index
            group = self.groups[self.current_index]
            inputs, targets = self.compute_inputs_targets(group)

        current_index = self.current_index + 1
        if current_index >= len(self.groups):
            current_index = current_index % (len(self.groups))
        self.current_index = current_index

        return inputs, targets        


    def compute_inputs_targets(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        if len(image_group) == 0:
            return None, None

        inputs = self.compute_inputs(image_group, annotations_group)
        targets = self.compute_targets(image_group)

        return inputs, targets


    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]


    def load_image(self, image_index):
        path = os.path.join(self.data_dir, "images", self.lines[image_index] + ".jpg")
        image = cv2.imread(path)
        return image


    def load_annotations_group(self, group):
        annotations_group = [self.load_annotations(annot_index) for annot_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert ('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert ('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group        


    def load_annotations(self, annot_index):
        filename = self.lines[annot_index] + ".xml"
        try:
            tree = ET.parse(os.path.join(self.data_dir, "annotations", filename))
            return self.parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)


    def parse_annotations(self, xml_root):
        annotations = {"labels" : np.empty((0,), dtype=np.int32), "bboxes" : np.empty((0, 4))}
        for i, element in enumerate(xml_root.iter("object")):
            try:
                box, label = self.read_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
            annotations['labels'] = np.concatenate([annotations['labels'], [label]])

        return annotations


    def read_annotation(self, element):
        class_name = findNode(element, "name").text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        bbox = np.zeros((4,))
        label = self.name_to_label(class_name)
        
        bndbox = findNode(element, "bndbox")
        bbox[0] = findNode(bndbox, "xmin", "bndbox.xmin", parse=float) # - 1
        bbox[1] = findNode(bndbox, "ymin", "bndbox.ymin", parse=float) # - 1
        bbox[2] = findNode(bndbox, "xmax", "bndbox.xmax", parse=float) # - 1
        bbox[3] = findNode(bndbox, "ymax", "bndbox.ymax", parse=float) # - 1

        return bbox, label


    def name_to_label(self, name):
        return self.classes.index(name)


    def filter_annotations(self, image_group, annotations_group, group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            invalid_indices = np.where((annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                                       (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                                       (annotations['bboxes'][:, 0] < 0) |
                                       (annotations['bboxes'][:, 1] < 0) |
                                       (annotations['bboxes'][:, 2] <= 0) |
                                       (annotations['bboxes'][:, 3] <= 0) |
                                       (annotations['bboxes'][:, 2] > image.shape[1]) |
                                       (annotations['bboxes'][:, 3] > image.shape[0]))[0]

            # # delete invalid indices
            # if len(invalid_indices):
            #     warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(group[index], image.shape, annotations['bboxes'][invalid_indices, :]))
            #     for k in annotations_group[index].keys():
            #         annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

            # if annotations['bboxes'].shape[0] == 0:
            #     warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(group[index], image.shape,))
        return image_group, annotations_group


    def preprocess_image(self, image, c, s, tgt_w, tgt_h):
        trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
        image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
        image = image.astype(np.float32)

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image


    def compute_inputs(self, image_group, annotations_group):
        batch_images = np.zeros((len(image_group), self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_hms = np.zeros((len(image_group), self.output_size, self.output_size, len(self.classes)), dtype=np.float32)
        batch_hms_2 = np.zeros((len(image_group), self.output_size, self.output_size, len(self.classes)), dtype=np.float32)
        batch_whs = np.zeros((len(image_group), self.max_detections, 2), dtype=np.float32)
        batch_regs = np.zeros((len(image_group), self.max_detections, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((len(image_group), self.max_detections), dtype=np.float32)
        batch_indices = np.zeros((len(image_group), self.max_detections), dtype=np.float32)

        for b, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0
            trans_input = get_affine_transform(c, s, self.input_shape[0])

            # inputs
            image = self.preprocess_image(image, c, s, tgt_w=self.input_shape[0], tgt_h=self.input_shape[1])
            batch_images[b] = image

            # outputs
            bboxes = annotations['bboxes']
            # assert bboxes.shape[0] != 0
            class_ids = annotations['labels']
            # assert class_ids.shape[0] != 0

            trans_output = get_affine_transform(c, s, self.output_size)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].copy()
                cls_id = class_ids[i]
                # (x1, y1)
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                # (x2, y2)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius_h, radius_w = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius_h = max(0, int(radius_h))
                    radius_w = max(0, int(radius_w))

                    radius = gaussian_radius_2((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius_h, radius_w)
                    draw_gaussian_2(batch_hms_2[b, :, :, cls_id], ct_int, radius)
                    batch_whs[b, i] = 1. * w, 1. * h
                    batch_indices[b, i] = ct_int[1] * self.output_size + ct_int[0]
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1

            return [batch_images, batch_hms_2, batch_whs, batch_regs, batch_reg_masks, batch_indices]

    def compute_targets(self, image_group):
        return np.zeros((len(image_group),))


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         output_size,
                         rot=0.,
                         inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list) and not isinstance(scale, tuple):
        scale = np.array([scale, scale], dtype=np.float32)

    if not isinstance(output_size, np.ndarray) and not isinstance(output_size, list) and not isinstance(output_size,
                                                                                                        tuple):
        output_size = np.array([output_size, output_size], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    src_h = scale_tmp[1]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def gaussian_radius(det_size, min_overlap=0.7):
    det_h, det_w = det_size
    rh = 0.1155 * det_h
    rw = 0.1155 * det_w
    return rh, rw


def gaussian_radius_2(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius_h, radius_w, k=1):
    diameter_h = 2 * radius_h + 1
    diameter_w = 2 * radius_w + 1
    gaussian = gaussian2D((diameter_h, diameter_w), sigma_w=diameter_w / 6, sigma_h=diameter_h / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_w), min(width - x, radius_w + 1)
    top, bottom = min(y, radius_h), min(height - y, radius_h + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_h - top:radius_h + bottom, radius_w - left:radius_w + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_gaussian_2(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D_2((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma_w=1, sigma_h=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-((x * x) / (2 * sigma_w * sigma_w) + (y * y) / (2 * sigma_h * sigma_h)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian2D_2(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


if __name__ == "__main__":
    df = pd.read_csv("/home/ubuntu/Datasets/COCO2017/Labels/labels.txt", sep=",", index_col=False, header=None)
    classes = df[0].to_list()
    dataset = DataGenerator("/home/ubuntu/Datasets/COCO2017", "train", classes, 1, (512, 512, 3), 100)

    print(dataset[0])