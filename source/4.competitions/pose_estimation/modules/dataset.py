import os
import copy
import cv2
import torch
import sys
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from modules.pose_utils import world2cam, cam2pixel
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode, input_shape, output_shape):

        self.data_dir = data_dir
        self.mode = mode
        self.joint_num = 24
        self.skeleton = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                    (21, 23))  # 인접된 관절 좌표 정의
        self.orig_img_shape = (1920, 1080)
        self.input_shape = (input_shape, input_shape) # 모델에 입력될 이미지 사이즈 (W, H)
        self.output_shape = (output_shape, output_shape, output_shape) # 모델 출력 Hitmap 사이즈 (W, H, D)
        self.db = self.data_lodaer()
        self.transform = transforms.Compose([transforms.CenterCrop(self.input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def data_lodaer(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '.pt')):
            db = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '.pt'))
        else:
            db = []
            for (roots, dirs, files) in os.walk(os.path.join(self.data_dir, self.mode, 'images')):
                if len(files) != 0:
                    camera_path = os.path.join(self.data_dir, self.mode, 'camera', roots.split('/')[-1] + '.json')
                    # print(camera_path)
                    camera = pd.read_json(camera_path)
                    f = [camera['intrinsics'][0][0], camera['intrinsics'][1][1]]  # focal length
                    c = [camera['intrinsics'][0][2], camera['intrinsics'][1][2]]  # principal point
                    t = np.array(camera['extrinsics'].tolist())[:, -1]
                    R = np.array(camera['extrinsics'].tolist())[:, :3]

                    for file in files:
                        annot_path = os.path.join(self.data_dir, self.mode, 'labels', roots.split('/')[-1][:-2], '3D_' + file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[3][:-3] + 'json')
                        annot = pd.read_json(annot_path)
                        joint_world = np.array(annot['annotations']['3d_pos']).squeeze(axis=-1)
                        joint_cam = world2cam(joint_world, R, t)
                        scale_factor_of_intrinsic_param = self.orig_img_shape[0]
                        joint_img = cam2pixel(joint_cam, f, c, scale_factor_of_intrinsic_param)
                        joint_vis = np.ones((self.joint_num, 1))
                        db.append({'img_path': os.path.join(roots, file),
                                     'joint_img': joint_img,
                                     'joint_vis': joint_vis,
                                     'joint_world': joint_world,
                                     'f': f,
                                     'c': c,
                                     't': t,
                                     'R': R})

                        if len(db) == 10:
                            torch.save(db, os.path.join(self.data_dir, self.mode, self.mode + '.pt'))
                            self.depth_max = max([np.max(np.abs(d['joint_img'][:, 2])) for d in db])
                            return db
            
        #     print(len(db))
        #     torch.save(db, os.path.join(self.data_dir, self.mode, self.mode + '.pt'))

        # self.depth_max = max([np.max(np.abs(d['joint_img'][:, 2])) for d in db])
        # return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])
        joint_img = data['joint_img']
        joint_vis = data['joint_vis']

        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])

        # 2. crop patch from img & generate patch joint ground truth
        img_patch = self.transform(Image.fromarray(cvimg))
        joint_img[:, 0] = joint_img[:, 0] - (self.orig_img_shape[0] - self.input_shape[0]) / 2
        joint_img[:, 1] = joint_img[:, 1] - (self.orig_img_shape[1] - self.input_shape[1]) / 2
        joint_img[:, 2] /= self.depth_max # 0~1 normalize
        joint_img[:, 2] = (joint_img[:, 2] + 1.0) / 2.
        for i in range(self.joint_num):
            joint_vis[i] *= (
                    (joint_img[i, 0] >= 0) & \
                    (joint_img[i, 0] < self.input_shape[1]) & \
                    (joint_img[i, 1] >= 0) & \
                    (joint_img[i, 1] < self.input_shape[0]) & \
                    (joint_img[i, 2] >= 0) & \
                    (joint_img[i, 2] < 1)
            )

        # 3. change coordinates to output space
        joint_img[:, 0] = joint_img[:, 0] / self.input_shape[0] * self.output_shape[0]
        joint_img[:, 1] = joint_img[:, 1] / self.input_shape[1] * self.output_shape[1]
        joint_img[:, 2] = joint_img[:, 2] * self.output_shape[2]

        joint_img = joint_img.astype(np.float32)
        joint_vis = (joint_vis > 0).astype(np.float32)

        return img_patch, joint_img, joint_vis



