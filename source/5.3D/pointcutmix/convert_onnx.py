from __future__ import print_function
import argparse, os, csv, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import onnxruntime, onnx
from torch.utils.data import Dataset, DataLoader
from data_utils.data_util import PointcloudScaleAndTranslate
from data_utils.ModelNetDataLoader import ModelNetDataLoader

from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2 import PointNet2ClsMsg
from models.dgcnn import DGCNN
from models.pointcnn import PointCNNCls

from utils import progress_bar, log_row

sys.path.append("./emd/")
import emd_module as emd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = '/data/datasets/modelnet40_normal_resampled/'
TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test', normal_channel=False)
test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()
print('======> Successfully loaded!')

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

criterion = cal_loss
correct = 0
total = 0

model = torch.load('onnx/model.pt')
model.to(device)
model.eval()

for j, data in enumerate(test_loader, 0):
    points, label = data
    points, label = points.to(device), label.to(device)[:, 0]
    points = points.transpose(2, 1)  # to be shape batch_size*3*N
    
    if j == 0:
        torch.onnx.export(model, points, 'onnx/convert_model.onnx')

    pred = model(points)
    loss = criterion(pred, label.long())

    pred_choice = pred.data.max(1)[1]
    correct += pred_choice.eq(label.data).cpu().sum()
    total += label.size(0)
    progress_bar(j, len(test_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (loss.item() / (j + 1), 100. * correct.item() / total, correct, total))

print(loss.item() / (j + 1), 100. * correct.item() / total)