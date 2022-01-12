import cv2
import json
import datetime
import torch
import pytorch_model_summary
import torchsummary
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from model.model import get_pose_net
from PIL import Image
from tqdm import tqdm
from modules.pose_utils import world2cam, cam2pixel

if __name__ == "__main__":
    joint_num = 24
    RESNET_TYPE = 18
    INPUT_SHAPE = (800, 800)
    OUTPUT_SHAPE = 200
    CKPT_PATH = "/data/pose_estimation/results/backup/POSENET_20210629091405/best.pt"
    SM_PATH = "/data/pose_estimation/DATA/task04_test/sample_submission.json"
    IMG_PATH = "/data/pose_estimation/DATA/task04_test/images"
    CAM_PATH = "/data/pose_estimation/DATA/task04_test/camera"

    transform = transforms.Compose([transforms.CenterCrop(INPUT_SHAPE), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_pose_net(RESNET_TYPE, OUTPUT_SHAPE, True, joint_num).to(device)
    model.load_state_dict(torch.load(CKPT_PATH), strict=False)
    # model.to(device)
    model.eval()

    # print(pytorch_model_summary.summary(model, torch.zeros(1, 3, 800, 800).to(device), show_input=True))
    torchsummary.summary(model, (3, 800, 800), device='cuda')

    with open(SM_PATH, encoding='utf-8', mode='r') as f:
        sm_arr = json.load(f)

    results = []
    for idx in  tqdm(range((len(sm_arr)))):
        file_name, joints = sm_arr[idx]['file_name'], sm_arr[idx]['3d_pos']
        # print(file_name, joints)

        folder_name = file_name.split('_')[:3]
        folder_name = '_'.join(folder_name)
        
        test_image = cv2.imread(f"{IMG_PATH}/{folder_name}/{file_name}", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        test_image = transform(Image.fromarray(test_image))
        test_image = torch.unsqueeze(test_image, 0)
        test_image = test_image.to(device)
        
        pred = model(test_image)
        # print(pred)
        # break
        
        for p_joints in pred[0]:
            p_joints = p_joints.cpu().data.numpy()
            joints.append([float(p_joints[0]), float(p_joints[1]), float(p_joints[2])])
        
        sm_arr[idx]['3d_pos'] = joints

        # print(sm_arr[idx])
        # break
        
    LOG_TIME = datetime.datetime.now().strftime("%m%d_%H%M%S")
    with open(f'./{LOG_TIME}.json', 'w') as outfile:
        json.dump(sm_arr, outfile)