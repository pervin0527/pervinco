import os
import cv2
import json
import torch
import pathlib
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import albumentations as A
from model.model import get_pose_net
from PIL import Image
from tqdm import tqdm
from modules.vis import  visualization
from modules.pose_utils import world2cam, cam2pixel
from matplotlib import pyplot as plt

if __name__ == "__main__":
    CKPT_PATH = "results/train/POSENET_20210628192919/best.pt"
    DATA_ROOT = "DATA/task04_train"

    joint_num = 24
    input_shape = (800, 800)
    output_shape = (200, 200, 200)
    orig_img_shape = (1920, 1080)
    transform = transforms.Compose([transforms.CenterCrop(input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    images_folders = pathlib.Path(DATA_ROOT)
    images_folders = list(images_folders.glob('images/*'))
    images_folders = sorted([str(path) for path in images_folders])

    for folder in images_folders:
        camera_name = folder.split('/')[-1]
        # print(camera_name)
        camera_path = f"{DATA_ROOT}/camera/{camera_name}.json"
        camera = pd.read_json(camera_path)

        f = [camera['intrinsics'][0][0], camera['intrinsics'][1][1]]  # focal length
        c = [camera['intrinsics'][0][2], camera['intrinsics'][1][2]]  # principal point
        t = np.array(camera['extrinsics'].tolist())[:, -1]
        R = np.array(camera['extrinsics'].tolist())[:, :3]

        images = pathlib.Path(folder)
        images = list(images.glob('*.jpg'))
        images = [str(path) for path in images]
        
        for image in images[:3]:
            image_file_name = image.split('/')[-1]
            # print(image_file_name)
            cvimg = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img_patch = transform(Image.fromarray(cvimg))
            img_patch = img_patch[0].numpy()

            label_folder_name = '_'.join(camera_name.split('_')[:-1])
            label_file_name = image_file_name.split('.')[:-1]
            label_file_name = label_file_name[0].split('_')
            del label_file_name[2]
            label_file_name = '_'.join(label_file_name)

            annot_path = f"{DATA_ROOT}/labels/{label_folder_name}/3D_{label_file_name}.json"
            annot = pd.read_json(annot_path)

            joint_world = np.array(annot['annotations']['3d_pos']).squeeze(axis=-1)
            joint_cam = world2cam(joint_world, R, t)
            scale_factor_of_intrinsic_param = orig_img_shape[0]
            joint_img = cam2pixel(joint_cam, f, c, scale_factor_of_intrinsic_param)
            joint_vis = np.ones((joint_num, 1))

            depth_max = np.max(np.abs(joint_img[:, 2]))

            joint_img[:, 0] = joint_img[:, 0] - (orig_img_shape[0] - input_shape[0]) / 2
            joint_img[:, 1] = joint_img[:, 1] - (orig_img_shape[1] - input_shape[1]) / 2
            joint_img[:, 2] /= depth_max # 0~1 normalize
            joint_img[:, 2] = (joint_img[:, 2] + 1.0) / 2.

            for i in range(joint_num):
                joint_vis[i] *= (
                        (joint_img[i, 0] >= 0) & \
                        (joint_img[i, 0] < input_shape[1]) & \
                        (joint_img[i, 1] >= 0) & \
                        (joint_img[i, 1] < input_shape[0]) & \
                        (joint_img[i, 2] >= 0) & \
                        (joint_img[i, 2] < 1)
                )

            # 3. change coordinates to output space
            joint_img[:, 0] = joint_img[:, 0] / input_shape[0] * output_shape[0]
            joint_img[:, 1] = joint_img[:, 1] / input_shape[1] * output_shape[1]
            joint_img[:, 2] = joint_img[:, 2] * output_shape[2]

            joint_img = joint_img.astype(np.float32)
            joint_vis = (joint_vis > 0).astype(np.float32)


            print(image_file_name)
            tmpimg = visualization(img_patch, joint_num, joint_img, joint_vis)
            tmpimg = cv2.cvtColor(tmpimg, cv2.COLOR_RGB2BGR)
            plt.imshow(tmpimg)
            plt.show()