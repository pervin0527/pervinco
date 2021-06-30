import os
import json
import random
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from modules.dataset import TestDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_json
import torch
from model.model import get_pose_net
from modules.pose_utils import pred2pixel, pixel2cam, cam2world

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = "/data/pose_estimation/DATA"
PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/predict_config.yml')
config = load_yaml(PREDICT_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# PREDICT
BATCH_SIZE = config['PREDICT']['batch_size']
INPUT_SHAPE = config['PREDICT']['input_shape']
OUTPUT_SHAPE = config['PREDICT']['output_shape']
RESNET_TYPE = config['PREDICT']['resnet_type']

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    orig_img_shape = (1920, 1080)

    TRAINED_MODEL_PATH = "/data/pose_estimation/results/train/POSENET_20210628192919/best.pt"
    SUBMISSION_PATH = './DATA/task04_test/sample_submission.json'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset & dataloader
    test_dataset = TestDataset(data_dir=DATA_DIR, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    depth_max = 5749.878616531615 #maximum depth value of train set

    # Load Model
    joint_num = 24
    model = get_pose_net(RESNET_TYPE, OUTPUT_SHAPE, True, joint_num).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'], strict=False)
    submission = pd.read_json(SUBMISSION_PATH)

    with open(SUBMISSION_PATH, encoding='utf-8', mode='r') as f:
        sm_arr = json.load(f)

    #Set trainer
    trainer = Trainer(model, device, None, None, None, None)
    model.eval()
    with torch.no_grad():
        for batch_index, (img_patch, img_path, f, c, t, R) in enumerate(test_dataloader):
            os.system("clear")
            print(len(test_dataloader) - batch_index)

            img_patch = img_patch.to(device)
            coord = model(img_patch)
            coord = coord.cpu()

            for i in range(BATCH_SIZE):
                pixel_coord = pred2pixel(coord[i], orig_img_shape, INPUT_SHAPE, OUTPUT_SHAPE, depth_max)
                cam_coord = pixel2cam(pixel_coord, f[i], c[i], 1920)
                world_coord = cam2world(torch.tensor(cam_coord), R[i], t[i])
                id = submission[submission['file_name'] == os.path.split(img_path[i])[1]].index[0]
                # sm_arr[id]['3d_pos'] = world_coord.tolist()

            sm_arr[id]['3d_pos'] = world_coord.tolist()

            # path_name = img_path[0].split('/')[-1]
            # sub_name = sm_arr[id]['file_name']
            # if sub_name != path_name:
            #     break
            #     print(img_path)
            
    LOG_TIME = datetime.datetime.now().strftime("%m%d_%H%M%S")
    with open(f'./{LOG_TIME}.json', 'w') as outfile:
        json.dump(sm_arr, outfile)