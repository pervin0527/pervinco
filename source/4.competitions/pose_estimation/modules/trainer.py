import os
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, model, device, metric_fn, optimizer=None, scheduler=None, logger=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.metric_fn = metric_fn

    def train_epoch(self, dataloader, epoch_index):
        self.model.train()
        self.train_total_loss = 0
        target_lst = []
        pred_lst = []
        
        idx = 1
        for batch_index, (img_patch, joint_img, joint_vis) in enumerate(tqdm(dataloader)):
            img_patch = img_patch.to(self.device)
            joint_img = joint_img.to(self.device)
            joint_vis = joint_vis.to(self.device)
            coord = self.model(img_patch)

            ## coordinate loss
            loss_coord = torch.abs(coord - joint_img) * joint_vis
            loss = loss_coord.mean()           
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.train_total_loss += loss
            self.train_mean_loss = self.train_total_loss / idx
            idx += 1

            target_lst.append(joint_img)
            pred_lst.append(coord)

        self.train_score = self.metric_fn(y_pred=pred_lst, y_answer=target_lst)
        msg = f'Epoch {epoch_index}, Train loss: {self.train_mean_loss}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)

        torch.save(model, './model/saved_model.pt')

    def validate_epoch(self, dataloader, epoch_index):
        self.model.eval()
        self.val_total_loss = 0
        target_lst = []
        pred_lst = []
        with torch.no_grad():
            for batch_index, (img_patch, joint_img, joint_vis) in enumerate(dataloader):
                img_patch = img_patch.to(self.device)
                joint_img = joint_img.to(self.device)
                joint_vis = joint_vis.to(self.device)
                coord = self.model(img_patch)
                ## coordinate loss
                loss_coord = torch.abs(coord - joint_img) * joint_vis
                loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1] + loss_coord[:, :, 2]) / 3.
                loss = loss_coord.mean()
                self.val_total_loss += loss
                # msg = f'Epoch {epoch_index}, Batch {batch_index}, Validation loss: {loss.item()}'
                # print(msg)
                target_lst.append(joint_img)
                pred_lst.append(coord)
                
            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = self.metric_fn(y_pred=pred_lst, y_answer=target_lst)
            msg = f'Epoch {epoch_index}, Validation loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)


