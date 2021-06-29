import os
import random
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from model.model import get_pose_net
from torch.utils.data import DataLoader
from modules.metrics import get_metric_fn
from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from torch.utils.data.sampler import SubsetRandomSampler

DEBUG = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'DATA')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, './config/train_config.yml')
config = load_yaml(TRAIN_CONFIG_PATH)

RANDOM_SEED = config['SEED']['random_seed']

EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
MODEL = config['TRAIN']['model']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']
INPUT_SHAPE = config['TRAIN']['input_shape']
OUTPUT_SHAPE = config['TRAIN']['output_shape']
RESNET_TYPE = config['TRAIN']['resnet_type']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{MODEL}_{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    make_directory(PERFORMANCE_RECORD_DIR)
    system_logger = get_logger(name='train', file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))
    
    # train_dataset = CustomDataset(data_dir=DATA_DIR, mode='task04_train', input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    # validation_dataset = CustomDataset(data_dir=DATA_DIR, mode='task04_train', input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset = CustomDataset(data_dir=DATA_DIR, mode='task04_train', input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    validation_split = .2
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
    
    print('Train set samples:',len(train_dataloader),  'Val set samples:', len(validation_dataloader))

    joint_num = 24
    model = get_pose_net(RESNET_TYPE, OUTPUT_SHAPE, True, joint_num).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 250, eta_min=1e-3)
    metric_fn = get_metric_fn

    trainer = Trainer(model, device, metric_fn, optimizer, scheduler, logger=system_logger)
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)

    criterion = 1E+8
    for epoch_index in range(EPOCHS):
        print(f"####### EPOCH {epoch_index+1} #######")
        trainer.train_epoch(train_dataloader, epoch_index)
        trainer.validate_epoch(validation_dataloader, epoch_index)

        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_mean_loss,
                                     validation_loss=trainer.val_mean_loss,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score
                                     )
        
        performance_recorder.save_performance_plot(final_epoch=epoch_index)

        early_stopper.check_early_stopping(loss=trainer.val_mean_loss)

        if early_stopper.stop:
            print('Early stopped')
            break

        if trainer.train_mean_loss < criterion:
            criterion = trainer.train_mean_loss
            performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'best.pt')
            performance_recorder.save_weight()
            torch.save(model, './results/saved_model.pt')
        
        print()