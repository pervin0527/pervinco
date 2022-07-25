import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from six import raise_from
from common import Generator


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


class DataGenerator(Generator):
    def __init__(self, data_dir, set_name, classes, image_extension='.jpg', skip_truncated=False, skip_difficult=False, **kwargs):

        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        # self.image_names = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_names = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, set_name + '.txt')).readlines()]
        self.image_extension = image_extension
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        
        self.labels = {}
        # for key, value in self.classes.items():
        #     self.labels[value] = key
        for index, label in enumerate(classes):
            self.labels[index] = label

        super(DataGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def has_label(self, label):
        return label in self.labels

    def has_name(self, name):
        return name in self.classes

    def name_to_label(self, name):
        return self.classes.index(name)

    def num_classes(self):
        return len(self.classes)

    def image_aspect_ratio(self, image_index):
        # path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        path = os.path.join(self.data_dir, 'images', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        # path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        path = os.path.join(self.data_dir, 'images', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        return image

    def __parse_annotation(self, element):
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

        return box, label

    def __parse_annotations(self, xml_root):
        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4))}
                       
        for i, element in enumerate(xml_root.iter('object')):
            try:
                box, label = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
            annotations['labels'] = np.concatenate([annotations['labels'], [label]])

        return annotations

    def load_annotations(self, image_index):
        filename = self.image_names[image_index] + '.xml'
        try:
            # tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            tree = ET.parse(os.path.join(self.data_dir, 'annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)


if __name__ == '__main__':
    generator = DataGenerator(
        '/data/Datasets/300VW_Dataset_2015_12_14/face_detection/train_512',
        'list',
        classes={"face" : 0},
        skip_difficult=True,
        batch_size=1
    )
    for inputs, targets in generator:
        print('hi')
