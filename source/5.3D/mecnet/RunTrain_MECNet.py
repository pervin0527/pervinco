#-*- coding:utf-8 -*-

import sys
import os.path as osp

sys.path.insert(0, '../')

from train_cls import train
from loss import MecNetLoss, MecNetANGLoss # for pos+ang: MecNetANGLoss
from models.mecnet import MECNet, MECANGNet # for pos+ang: MECANGNET
from models import mobilenet
from lsar import LSAR_POS_SOFT, LSAR_POSANG_SOFT # for pos+ang: LSAR_POSANG_SOFT

import os.path as osp
import os
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models
from torch.nn import DataParallel
import torch.optim as optim


## args
class MECNetArgs():
    def __init__(self):

        self.dataset = 'LSAR_POS_SOFT' # 'LSAR_POSANG_SOFT' for pos+angle
        self.model_name = 'menet'
        self.learn_alpha = True

        self.expflag = 'test'
        self.device = '0'
        self.maxflr = 3
        self.inflr = 1
        self.flrflag = True

        self.resume = False
        self.weights = None #'results/test/epoch_005.pth.tar'

        self.datapath = osp.join('..', 'traintest_dataset_12_f1') # dataset path

## parameters
class MECNetConfig():
    def __init__(self):

        self.n_epochs = 15
        self.batch_size = 12
        self.seed = 7
        self.shuffle = True
        self.num_workers = 6
        self.snapshot = 5
        self.val_freq = 1
        self.max_grad_norm = 0
        self.cuda = True

        self.lr = 1e-4
        self.weight_decay = 0.0005
        self.lr_decay = 0.1
        self.lr_stepvalues = [10]
        self.print_freq = 20
        self.logdir = './results'
        self.expdir = './results'


args = MECNetArgs()
cfg = MECNetConfig()

alpha = -3.0
dropout = 0.5
color_jitter = 0.7

alphap=alpha
alphaa=alpha
alphaf=alpha

# 학습 checkpoint 저장 경로 만들기
savedir = osp.join(cfg.logdir, args.expflag)
if not osp.isdir(savedir):
    os.makedirs(savedir)
cfg.expdir = osp.join(cfg.logdir, args.expflag)

data_dir = args.datapath
stats_file = osp.join(data_dir, 'stats.txt') # Tensor normalize를 위한 파라미터 값.
expflag = args.expflag

stats = np.loadtxt(stats_file)
# transformers 이미지 전처리
tforms = [transforms.Resize(256)]
if color_jitter > 0:
    assert color_jitter <= 1.0 # 1.0 보다 크거나 같으면 AssertError
    print('ColorJitter data augmentation')
	# https://nrhan.tistory.com/entry/Data-augmentation-color-jitter
    tforms.append(transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.5)) # torchvision transform - color jitter추가.

tforms.append(transforms.ToTensor()) # 텐서 변환
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))) # normalization
data_transform = transforms.Compose(tforms)

# data loader
kwargs = dict(data_path=data_dir, train=True, transform=data_transform, seed=cfg.seed, maxflr=args.maxflr, inflr=args.inflr, flrflag=args.flrflag)
if args.dataset.find('ANG')>=0:
    print("ANG")
    data_set = LSAR_POSANG_SOFT(**kwargs)
else:
    print("MECNET")
    data_set = LSAR_POS_SOFT(**kwargs) # keyword argument dict({'키워드' : '특정 값'})

# load num classes
numcls = data_set.num_classes
numang = data_set.num_angles
numflr = data_set.num_floors

# loss function
kwargs = dict(learn_alpha=args.learn_alpha, alphap=alphap, alphaa=alphaa, alphaf=alphaf, num_cls=numcls, num_ang=numang, num_flr=numflr )

if args.dataset.find('ANG')>=0:
    criterion = MecNetANGLoss(**kwargs)
else:
    criterion = MecNetLoss(**kwargs)


# model
feature_extractor = mobilenet.__dict__['mobilenetv2'](pretrained=True)
if args.dataset.find('ANG')>=0:
    model = MECANGNet(feature_extractor, droprate=dropout, pretrained=True,
          num_classes=numcls,num_angles=numang,num_floors=numflr)
else:
    model = MECNet(feature_extractor, droprate=dropout, pretrained=True, num_classes=numcls, num_angles=numang, num_floors=numflr)


# optimizer
param_list = [{'params': model.parameters()}]
param_list.append({'params': [criterion.alphap, criterion.alphaa, criterion.alphaf]})
optimizer = optim.Adam(param_list, lr=cfg.lr, weight_decay=cfg.weight_decay)


# trainer
exp = '{:s}_{:s}'.format(args.dataset, args.expflag)
train(model, optimizer, criterion, data_set, cfg, exp, weights=args.weights, resume=args.resume)
