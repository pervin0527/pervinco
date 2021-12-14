import cv2
from src.utils import read_xml, write_xml

class Dataset:
    def __init__(self, images, annotations, classes, **kwargs):
        self.__dict__ = kwargs
        self.images = images
        self.annotations = annotations
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = self.images[index]
        annot_file = self.annotations[index]

        image = cv2.imread(img_file)
        bboxes, labels = read_xml(annot_file, self.classes, format='albumentations')

        return {"img_file" : img_file,
                "annot_file" : annot_file,
                "image" : image,
                "bboxes" : bboxes,
                "labels" : labels}

class Augmentations:
    def __init__(self, dataset, augmentations):
        self.dataset = dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data = self.augmentations(**data)
        return data