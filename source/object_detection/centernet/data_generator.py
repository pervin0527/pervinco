import os
import cv2
import math
import random
import warnings
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

from six import raise_from
from data_utils import affine_transform, get_affine_transform, gaussian_radius, gaussian_radius_2, draw_gaussian, draw_gaussian_2


def _findNode(parent, name, debug_name=None, parse=None):
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


class Datasets(tf.keras.utils.Sequence):
    def __init__(self, data_dir, classes, batch_size=1, input_size=512, max_detections=100, shuffle=False):
        self.data_dir = data_dir
        self.classes = classes
        self.images_names = [l.strip().split(None, 1)[0] for l in open(f"{self.data_dir}/list.txt").readlines()]

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = self.input_size // 4
        self.max_detections = max_detections
        
        self.groups = None
        self.shuffle_groups = shuffle
        self.current_index = 0

        self.labels = {}
        # for key, value in self.classes.items():
        #     self.labels[value] = key
        for index, label in enumerate(classes):
            self.labels[index] = label

        self.group_images()

    def __len__(self):
        return len(self.groups)

    def size(self):
        return len(self.images_names)

    def name_to_label(self, name):
        return self.classes.index(name)

    def num_classes(self):
        return len(self.classes)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)
        self.current_index = 0


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

            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(group[index], image.shape, annotations['bboxes'][invalid_indices, :]))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

            if annotations['bboxes'].shape[0] == 0:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(group[index], image.shape,))

        return image_group, annotations_group


    def load_image(self, image_index):
        path = f"{self.data_dir}/images/{self.images_names[image_index]}.jpg"
        image = cv2.imread(path)

        return image


    def __parse_annotation(self, element):
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float)
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float)
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float)
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float)

        return truncated, difficult, box, label


    def __parse_annotations(self, xml_root):
        annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
            annotations['labels'] = np.concatenate([annotations['labels'], [label]])

        return annotations


    def load_annotations(self, image_index):
        filename = f"{self.data_dir}/annotations/{self.images_names[image_index]}.xml"
        try:
            tree = ET.parse(filename)
            return self.__parse_annotations(tree.getroot())

        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)

        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)


    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]


    def load_annotations_group(self, group):
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert ('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert ('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group


    def compute_targets(self, image_group, annotations_group):
        return np.zeros((len(image_group), ))


    def compute_inputs(self, image_group, annotations_group):
        batch_images = np.zeros((len(image_group), self.input_size, self.input_size, 3), dtype=np.float32)
        batch_hms = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()), dtype=np.float32)
        batch_hms_2 = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()), dtype=np.float32)
        batch_whs = np.zeros((len(image_group), self.max_detections, 2), dtype=np.float32)
        batch_regs = np.zeros((len(image_group), self.max_detections, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((len(image_group), self.max_detections), dtype=np.float32)
        batch_indices = np.zeros((len(image_group), self.max_detections), dtype=np.float32)

        for b, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0
            image = (image/ 127.5) - 1
            batch_images[b] = image

            bboxes = annotations['bboxes']
            assert bboxes.shape[0] != 0
            class_ids = annotations['labels']
            assert class_ids.shape[0] != 0

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


    def compute_inputs_targets(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        if len(image_group) == 0:
            return None, None

        inputs = self.compute_inputs(image_group, annotations_group)
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def group_images(self):
        order = list(range(self.size()))
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


if __name__ == "__main__":
    dataset = Datasets("/data/Datasets/WIDER/CUSTOM_XML/train", ["face"], 1, 512, 10, False)
    for inputs, targets in dataset:
        for input_data in inputs:
            print(input_data.shape)
                    
        image = inputs[0][0]
        heatmap = inputs[1][0]
        print(image.shape, heatmap.shape)

        cv2.imshow("image", image)
        cv2.imshow("heatmap", heatmap)
        cv2.waitKey(0)