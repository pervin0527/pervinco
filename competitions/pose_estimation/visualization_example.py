import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.vis import  visualization
from modules.pose_utils import cam2pixel, world2cam


if __name__ == '__main__':
    # Arguments 정의
    joint_num = 24
    skeleton = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                (21, 23)) # 인접된 관절 좌표 정의

    # 카메라 파라미터 로드
    camera_param = pd.read_json('./DATA/task04_train/camera/30_F160A_1.json')
    f = [camera_param['intrinsics'][0][0], camera_param['intrinsics'][1][1]]  # focal length
    c = [camera_param['intrinsics'][0][2], camera_param['intrinsics'][1][2]]  # principal point
    t = np.array(camera_param['extrinsics'].tolist())[:, -1] # extrinsic params
    R = np.array(camera_param['extrinsics'].tolist())[:, :3] # extrinsic params

    # 칼리브레이션 및 시각화
    scaling_factor = 1920
    image = cv2.imread('./DATA/task04_train/images/30_F160A_1/30_F160A_1_4.jpg', cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    annot = pd.read_json('./DATA/task04_train/labels/30_F160A/3D_30_F160A_4.json')
    joint_world = np.array(annot['annotations']['3d_pos']).squeeze(axis=-1)
    joint_cam = world2cam(joint_world, R, t)
    joint_img = cam2pixel(joint_cam, f, c, scaling_factor)
    joint_vis = np.ones((joint_num, 1))
    tmpimg = visualization(image, joint_num, joint_img, joint_vis)
    tmpimg = cv2.cvtColor(tmpimg, cv2.COLOR_RGB2BGR)
    plt.imshow(tmpimg)
    plt.show()

    print(joint_img.shape)
    print(joint_img)

    # cv2.imshow('result', np.uint8(tmpimg))
    # cv2.waitKey(0)