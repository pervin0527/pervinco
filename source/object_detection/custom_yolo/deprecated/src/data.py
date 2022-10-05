import cv2
import numpy as np
import xml.etree.ElementTree as ET
from copy import copy
from sklearn.model_selection import KFold as KF, StratifiedKFold as SKF


class BaseDataset:
    def __init__(self, images, labels, classes, **kwargs):
        self.__dict__ = kwargs
        self.images = images
        self.labels = labels
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'image': self.images[index], 'label': self.labels[index]}


def base_inheritance(cls):
    def inherit_base(dataset, *args, **kwargs):
        return type(cls.__name__, (cls, dataset.__class__.mro()[-2],), {})(dataset, *args, **kwargs)
    return inherit_base


@base_inheritance
class LoadImages:
    def __init__(self, dataset, color_mode=1):
        # color_mode 1: "color", 0: "grey", -1: "unchanged"
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.color_mode = {1: "color", 0: "grey", -1: "unchanged"}.get(color_mode, color_mode)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = cv2.imread(
            sample["image"],
            flags={
                "color": cv2.IMREAD_COLOR,
                "grey": cv2.IMREAD_GRAYSCALE,
                "unchanged": cv2.IMREAD_UNCHANGED
            }[self.color_mode]
        )
        if sample["image"].ndim == 2:
            sample["image"] = sample["image"][..., np.newaxis]
        return sample


@base_inheritance
class ResizeImages:
    def __init__(self, dataset, image_size=(200, 200), interpolation=cv2.INTER_LINEAR):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.image_size = image_size
        self.interpolation = interpolation

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = cv2.resize(sample["image"], dsize=self.image_size, interpolation=self.interpolation)
        return sample


@base_inheritance
class OneHotLabels:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["label"] = np.array(list(map(lambda x: int(x==sample["label"]), self.classes)))
        return sample


@base_inheritance
class SparseLabels:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["label"] = self.classes.index(sample["label"])
        return sample


@base_inheritance
class Augmentations:
    def __init__(self, dataset, augmentations):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.augmentations = augmentations

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = self.augmentations(**sample)
        return sample


@base_inheritance
class Transforms:
    def __init__(self, dataset, transforms):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = self.transforms(sample)
        return sample


class KFold:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def split(self, n_splits=5, info=False):
        kf = KF(n_splits=n_splits, shuffle=True).split(self.images, self.labels)
        folds = []
        for train_idx, test_idx in kf:
            train_dataset = copy(self.dataset)
            train_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], train_idx))
            train_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], train_idx))

            test_dataset = copy(self.dataset)
            test_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], test_idx))
            test_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], test_idx))

            folds.append((train_dataset, test_dataset))

        if info:
            folds_info(folds)
        return folds


class StratifiedKFold:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def split(self, n_splits=5, info=False):
        skf = SKF(n_splits=n_splits, shuffle=True).split(self.images, self.labels)
        folds = []
        for train_idx, test_idx in skf:
            train_dataset = copy(self.dataset)
            train_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], train_idx))
            train_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], train_idx))

            test_dataset = copy(self.dataset)
            test_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], test_idx))
            test_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], test_idx))

            folds.append((train_dataset, test_dataset))

        if info:
            folds_info(folds)
        return folds

def folds_info(folds, **kwargs):
    if folds:
        kwargs = {"x": [], "y": [[], []]}
        for i, (train_dataset, test_dataset) in enumerate(folds):
            kwargs["x"].append(i+1)
            train_data_size = 0
            test_data_size = 0
            for _ in train_dataset.images: train_data_size += 1
            for _ in test_dataset.images: test_data_size += 1
            kwargs["y"][0].append(train_data_size)
            kwargs["y"][1].append(test_data_size)
        for fold, train_data_size, test_data_size in zip(kwargs["x"], *kwargs["y"]):
            print(f"[fold {fold}] train_data_size: {train_data_size}, test_data_size: {test_data_size}")
    stacked_bar(**kwargs)


@base_inheritance
class LoadPascalVOCLabels:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.bbox_format = "albumentations"

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['labels'], sample['bboxes'] = self.__decode_xml(sample.pop('label'))
        return sample

    def __decode_xml(self, label):
        tree = ET.parse(label)
        root = tree.getroot()
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        objects = root.findall('object')
        labels, bboxes = [], []
        for obj in objects:
            labels.append(obj.find('name').text)
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text) / width
            ymin = int(box.find('ymin').text) / height
            xmax = int(box.find('xmax').text) / width
            ymax = int(box.find('ymax').text) / height
            bboxes.append([xmin, ymin, xmax, ymax])
        return labels, bboxes