import cv2
import math
import random
import warnings
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from data_utils import findNode, affine_transform, get_affine_transform, gaussian_radius, draw_gaussian, gaussian_radius_2, draw_gaussian_2

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, classes, batch_size, input_shape, max_detections):
        self.data_dir = data_dir
        self.classes = classes
        self.input_shape = input_shape
        self.batch_size = int(batch_size)
        self.max_detections = int(max_detections)

        self.file_names = open(f"{data_dir}/list.txt", "r").readlines()
        self.groups = None
        self.current_index = 0
        self.group_images()

    def __len__(self):
        return len(self.file_names)

    def size(self):
        return len(self.file_names)

    def group_images(self):
        order = list(range(self.size()))
        random.shuffle(order)

        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def __getitem__(self, index):
        group = self.groups[self.current_index]
        inputs, targets = self.compute_input_targets(group)

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

    def compute_input_targets(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)
        inputs = self.compute_inputs(image_group, annotations_group)
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets
        
    def load_image_group(self, group):
        return [self.load_image(index) for index in group]

    def load_image(self, index):
        file_name = self.file_names[index].strip()
        path = f"{self.data_dir}/images/{file_name}.jpg"
        image = cv2.imread(path)
        return image

    def load_annotations_group(self, group):
        annotations_group = [self.load_annotations(index) for index in group]
        return annotations_group

    def load_annotations(self, index):
        file_name = self.file_names[index].strip()
        path = f"{self.data_dir}/annotations/{file_name}.xml"
        tree = ET.parse(path)
        return self.parse_annotations(tree.getroot())

    def parse_annotations(self, xml_root):
        annotations = {"labels" : np.empty((0, ), dtype=np.int32), "bboxes" : np.empty((0, 4))}
        for i, element in enumerate(xml_root.iter("object")):
            box, label = self.parse_annotation(element)
            annotations["bboxes"] = np.concatenate([annotations["bboxes"], [box]])
            annotations["labels"] = np.concatenate([annotations["labels"], [label]])

        return annotations

    def parse_annotation(self, element):
        class_name = findNode(element, "name").text

        box = np.zeros((4, ))
        label = self.classes.index(class_name)

        bounding_box = findNode(element, "bndbox")
        box[0] = float(findNode(bounding_box, "xmin").text)
        box[1] = float(findNode(bounding_box, "ymin").text)
        box[2] = float(findNode(bounding_box, "xmax").text)
        box[3] = float(findNode(bounding_box, "ymax").text)

        return box, label

    def filter_annotations(self, image_group, annotations_group, group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            if annotations['bboxes'].shape[0] == 0:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
                    group[index],
                    image.shape,
                ))
        return image_group, annotations_group


    def compute_inputs(self, image_group, annotations_group):
        batch_images = np.zeros((len(image_group), self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_hms = np.zeros((len(image_group), self.input_shape[0] // 4, self.input_shape[0] // 4, len(self.classes)), dtype=np.float32)
        batch_hms_2 = np.zeros((len(image_group), self.input_shape[0] // 4, self.input_shape[0] // 4, len(self.classes)), dtype=np.float32)
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
            image = (image / 128.0) - 1
            batch_images[b] = image

            # outputs
            bboxes = annotations['bboxes']
            assert bboxes.shape[0] != 0
            class_ids = annotations['labels']
            assert class_ids.shape[0] != 0

            trans_output = get_affine_transform(c, s, self.input_shape[0] // 4)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].copy()
                cls_id = class_ids[i]
                # (x1, y1)
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                # (x2, y2)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.input_shape[0] // 4 - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.input_shape[0] // 4 - 1)
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
                    batch_indices[b, i] = ct_int[1] * self.input_shape[0] // 4 + ct_int[0]
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1

        return [batch_images, batch_hms_2, batch_whs, batch_regs, batch_reg_masks, batch_indices]


    def compute_targets(self, image_group, annotations_group):
        return np.zeros((len(image_group),))


    def preprocess_image(self, image, c, s, tgt_w, tgt_h):
        trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
        image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
        image = image.astype(np.float32)

        # image[..., 0] -= 103.939
        # image[..., 1] -= 116.779
        # image[..., 2] -= 123.68

        return image



if __name__ == "__main__":
    train_data_dir = "/data/Datasets/WIDER/FACE/train_512"
    train_generator = DataGenerator(train_data_dir, ["face"], 32, (512, 512, 3), 10)
    data = train_generator[0]
    print(data)