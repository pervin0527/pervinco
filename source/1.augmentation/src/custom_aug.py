import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy


class CutMix(A.BasicTransform):
    def __init__(
        self,
        dataset,
        bbox_width_range=(0.3, 0.5),
        bbox_height_range=(0.3, 0.5),
        pad_size=(0.1, 0.1),
        mix_label=True,
        always_apply=False,
        p=1
    ):
        super(CutMix, self).__init__(always_apply, p)
        self.dataset = dataset
        self.sub_sample = None
        self.bbox_width_range = bbox_width_range
        self.bbox_height_range = bbox_height_range
        self.pad_size = pad_size
        self.r = 0.
        self.mix_label = mix_label

    def image_apply(self, image, **kwargs):
        index = random.randrange(self.dataset.__len__())
        self.sub_sample = self.dataset[index]
        bx, by, bw, bh, self.r = rand_bbox(image.shape, self.bbox_width_range, self.bbox_height_range, self.pad_size)
        image[by:by+bh, bx:bx+bw, :] = self.sub_sample["image"][by:by+bh, bx:bx+bw, :]
        return image

    def label_apply(self, label, **kwargs):
        if self.mix_label:
            label = label * (1 - self.r) + self.sub_sample["label"] * self.r
        return label

    @property
    def targets(self):
        return {"image": self.image_apply, "label": self.label_apply}

def rand_bbox(image_shape, bbox_width_range, bbox_height_range, pad_size):
    # shape: (h, w, c)
    W = image_shape[1]
    H = image_shape[0]
    if str(type(bbox_width_range[0])) == "<class 'float'>":
        bw = int(W * random.uniform(*bbox_width_range) + 0.5)
    else:
        bw = random.randrange(*bbox_width_range)
    if str(type(bbox_height_range[0])) == "<class 'float'>":
        bh = int(H * random.uniform(*bbox_height_range) + 0.5)
    else:
        bh = random.randrange(*bbox_height_range)
    if str(type(pad_size[0])) == "<class 'float'>":
        pw = int(W * pad_size[0] + 0.5)
        ph = int(H * pad_size[1] + 0.5)
    else:
        pw, ph = pad_size
    bx = random.randrange(pw, W - pw - bw)
    by = random.randrange(ph, H - ph - bh)
    r = bw * bh / (W * H)
    return bx, by, bw, bh, r


class MixUp(A.BasicTransform):
    def __init__(
        self,
        dataset,
        rate_range=(0.3, 0.5),
        mix_label=True,
        always_apply=False,
        p=1
    ):
        super(MixUp, self).__init__(always_apply, p)
        self.dataset = dataset
        self.sub_sample = None
        self.rate_range = rate_range
        self.r = 0.
        self.mix_label = mix_label

    def image_apply(self, image, **kwargs):
        index = random.randrange(self.dataset.__len__())
        self.sub_sample = self.dataset[index]
        resize = A.Compose([
            A.RandomSizedBBoxSafeCrop(*image.shape[:2])
            ], bbox_params=A.BboxParams(format='albumentations', min_area=0.3, min_visibility=0.3, label_fields=['labels'])
        )
        self.sub_sample = resize(**self.sub_sample)
        self.r = random.uniform(*self.rate_range)
        image = image * (1 - self.r) + self.sub_sample["image"] * self.r
        image = np.uint8(image)
        return image

    def label_apply(self, label, **kwargs):
        if self.mix_label:
            label = label * (1 - self.r) + self.sub_sample["label"] * self.r
        return label

    @property
    def targets(self):
        return {"image": self.image_apply, "label": self.label_apply}


class RandomLocateInFrame(A.BasicTransform):
    def __init__(
        self,
        h,
        w,
        interpolation=cv2.INTER_AREA,
        border_mode=cv2.BORDER_REPLICATE,
        always_apply=False,
        p=1,
    ):
        super(RandomLocateInFrame, self).__init__(always_apply, p)
        self.h = h
        self.w = w
        self.interpolation = interpolation
        self.border_mode = border_mode

    @property
    def target_dependence(self):
        return {'image': ['bboxes', 'labels'],
                'bboxes': ['image'],
                'labels': ['image', 'bboxes']}

    def crop_side_with_safe_bbox(self, frame_side, img_side, safe_set):
        side_min = random.choice(list(set(range(img_side - frame_side)) & safe_set))
        side_max = side_min + frame_side + 1
        while side_max < img_side:
            if side_max in safe_set:
                break
            side_max += 1
        return side_min, side_max
            
    def apply(self, image, **params):
        img_h, img_w, _ = image.shape

        h_set = set(range(img_h))
        w_set = set(range(img_w))

        for bbox in params['bboxes']:
            w_set -= set(range(int(img_w * bbox[0] + 0.5), int(img_w * bbox[2] + 0.5) + 1))
            h_set -= set(range(int(img_h * bbox[1] + 0.5), int(img_h * bbox[3] + 0.5) + 1))

        # crop
        if self.h < img_h:
            self.img_h_min, self.img_h_max = self.crop_side_with_safe_bbox(self.h, img_h, h_set)
        else:
            self.img_h_min, self.img_h_max = 0, img_h

        if self.w < img_w:
            self.img_w_min, self.img_w_max = self.crop_side_with_safe_bbox(self.w, img_w, w_set)
        else:
            self.img_w_min, self.img_w_max = 0, img_w
        
        img = image[self.img_h_min:self.img_h_max, self.img_w_min:self.img_w_max, ...]

        # resize and locate
        img_h, img_w, _ = img.shape
        if self.h / img_h < self.w / img_w:
            self.r = self.h / img_h
        else:
            self.r = self.w / img_w

        self.h_min = random.randrange(self.h - int(img_h * self.r + 0.5) + 1)
        self.w_min = random.randrange(self.w - int(img_w * self.r + 0.5) + 1)

        mtrx = np.array(
            [[self.r, 0, self.w_min],
             [0, self.r, self.h_min]]
        , dtype=np.float32)

        img = cv2.warpAffine(img, mtrx, (self.w, self.h), flags=self.interpolation, borderMode=self.border_mode)

        return img

    def apply_to_bboxes(self, bboxes, **params):
        new_bboxes = []
        img_h, img_w, _ = params['image'].shape
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[0] * img_w, bbox[1] * img_h, bbox[2] * img_w, bbox[3] * img_h
            if x_min >= self.img_w_min and y_min >= self.img_h_min and x_max < self.img_w_max and y_max < self.img_h_max:
                new_bbox = (
                    ((x_min - self.img_w_min) * self.r + self.w_min) / self.w,
                    ((y_min - self.img_h_min) * self.r + self.h_min) / self.h,
                    ((x_max - self.img_w_min) * self.r + self.w_min) / self.w,
                    ((y_max - self.img_h_min) * self.r + self.h_min) / self.h,
                    bbox[-1]
                    )
                new_bboxes.append(new_bbox)
        return new_bboxes

    def apply_to_labels(self, labels, **params):
        new_labels = []
        img_h, img_w, _ = params['image'].shape
        for bbox, label in zip(params['bboxes'], labels):
            x_min, y_min, x_max, y_max = bbox[0] * img_w, bbox[1] * img_h, bbox[2] * img_w, bbox[3] * img_h
            if x_min >= self.img_w_min and y_min >= self.img_h_min and x_max < self.img_w_max and y_max < self.img_h_max:
                new_labels.append(label)
        
        return new_labels

    @property
    def targets(self):
        return {'image': self.apply, 'bboxes': self.apply_to_bboxes, 'labels': self.apply_to_labels}


class Mosaic(A.BasicTransform):
    def __init__(
        self,
        dataset,
        height_split_range=(0.25, 0.75),
        width_split_range=(0.25, 0.75),
        transforms=[],
        bbox_params=None,
        always_apply=False,
        p=0.5,
    ):
        super(Mosaic, self).__init__(always_apply, p)
        self.dataset = dataset
        self.height_split_range = height_split_range
        self.width_split_range = width_split_range
        if bbox_params is None:
            bbox_params = A.BboxParams(format=dataset.bbox_format, min_area=0.3, min_visibility=0.3, label_fields=['labels'])
        self.transforms = transforms
        self.bbox_params = bbox_params

    @property
    def target_dependence(self):
        return {'image': ['bboxes', 'labels']}

    def get_piece(self, data, h, w):
        transforms = self.transforms + [RandomLocateInFrame(h, w, always_apply=True)]
        annotation = A.Compose(transforms, bbox_params=self.bbox_params)
        data = annotation(**data)
        return data

    def locate_bboxes(self, piece, h_r, w_r, h_p, w_p):
        def locate_bbox(bbox):
            return [w_r * w_p + bbox[0][0] * abs(w_p - w_r),
                    h_r * h_p + bbox[0][1] * abs(h_p - h_r),
                    w_r * w_p + bbox[0][2] * abs(w_p - w_r),
                    h_r * h_p + bbox[0][3] * abs(h_p - h_r),
                    bbox[1]]
        return list(map(locate_bbox, zip(piece['bboxes'], piece['labels'])))

    def apply(self, image, **params):
        height, width = params['rows'], params['cols']
        bboxes, labels = [], []
        for bbox in params['bboxes']:
            bboxes.append(list(bbox[:-1]))
            labels.append(bbox[-1])
        params['bboxes'] = bboxes
        params['labels'] = labels
        
        h_r = random.uniform(*self.height_split_range)
        w_r = random.uniform(*self.width_split_range)
        h = int(height * h_r + 0.5)
        w = int(width * w_r + 0.5)
        tl = self.get_piece({'image': image, **params}, h, w)
        tr = self.get_piece(self.dataset[random.randrange(self.dataset.__len__())], h, width - w)
        bl = self.get_piece(self.dataset[random.randrange(self.dataset.__len__())], height - h, w)
        br = self.get_piece(self.dataset[random.randrange(self.dataset.__len__())], height - h, width - w)
        t = np.concatenate((tl['image'], tr['image']), axis=1)
        b = np.concatenate((bl['image'], br['image']), axis=1)
        image = np.concatenate((t, b), axis=0)

        bboxes = self.locate_bboxes(tl, h_r, w_r, 0, 0)
        bboxes += self.locate_bboxes(tr, h_r, w_r, 0, 1)
        bboxes += self.locate_bboxes(bl, h_r, w_r, 1, 0)
        bboxes += self.locate_bboxes(br, h_r, w_r, 1, 1)
        self.bboxes = bboxes

        return image

    def apply_to_bboxes(self, bboxes, **params):
        return deepcopy(self.bboxes)

    @property
    def targets(self):
        return {'image': self.apply, 'bboxes': self.apply_to_bboxes}