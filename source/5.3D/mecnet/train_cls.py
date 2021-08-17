import sys
import os
import os.path as osp
import time
import numpy as np

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable
from utils import load_state_dict



def train(model, opt, criterion, train_dataset, cfg, exp, weights=None, resume=None):

    # activate GPUs
    if cfg.cuda:
        model.cuda()
        criterion.cuda()

    # set random seed
    torch.manual_seed(cfg.seed)
    if cfg.cuda:
        torch.cuda.manual_seed(cfg.seed)

    train = True
    start_epoch = int(0)

    # resume checkpoint
    if weights:
        if osp.isfile(weights):
            loc_func = lambda storage, loc: storage
            checkpoint = torch.load(weights, map_location=loc_func)
            load_state_dict(model, checkpoint['model_state_dict'])

            if resume:
                opt.load_state_dict(checkpoint['optim_state_dict'])
                start_epoch = checkpoint['epoch']
                if checkpoint.has_key('criterion_state_dict'):
                    c_state = checkpoint['criterion_state_dict']
                    append_dict = {k: torch.Tensor([0.0])
                                   for k,_ in criterion.named_parameters()
                                   if not k in c_state}
                    c_state.update(append_dict)
                    criterion.load_state_dict(c_state)
            print('Loaded checkpoint {:s} epoch {:d}'.format(weights, checkpoint['epoch']))


    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=cfg.batch_size, shuffle=cfg.shuffle,
            num_workers=cfg.num_workers, pin_memory=True)


    # start training
    for epoch in range(start_epoch, cfg.n_epochs):

        # SAVE CHECKPOINT
        if epoch % cfg.snapshot == 0:
            filename = osp.join(cfg.expdir, 'epoch_{:03d}.pth.tar'.format(epoch))
            cp_dict = \
              {'epoch': epoch, 'model_state_dict': model.state_dict(),
               'optim_state_dict': opt.state_dict(),
               'criterion_state_dict': criterion.state_dict()}
            torch.save(cp_dict, filename)
            print('Save checkpoint {}'.format(epoch))

        # ADJUST LR
        decay_factor = 1
        for s in cfg.lr_stepvalues:
            if epoch < s:
                break
            decay_factor *= cfg.lr_decay
        lr = cfg.lr * decay_factor
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data_var = Variable(data, requires_grad=train)
            if cfg.cuda:
                data_var = data_var.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            target_var = Variable(target, requires_grad=False)

            with torch.set_grad_enabled(train):
                output = model(data_var)
                cur_loss = criterion(output.unsqueeze(1), target_var.unsqueeze(1))
                opt.zero_grad()
                cur_loss.backward()
                opt.step()

                loss = cur_loss.item()

            # PRINT
            if batch_idx % cfg.print_freq == 0:
              n_iter = epoch*len(train_loader) + batch_idx
              epoch_count = float(n_iter)/len(train_loader)
              print('[{:s}]\t Epoch {:d}\t Iter {:d}/{:d}\t Loss {:f}\t lr: {:f}'.\
                format(exp, epoch, batch_idx, len(train_loader)-1, loss, lr))


    # Save final checkpoint
    cp_dict = \
      {'epoch': epoch, 'model_state_dict': model.state_dict(),
       'optim_state_dict': opt.state_dict(),
       'criterion_state_dict': criterion.state_dict()}
    filename = osp.join(cfg.expdir, 'final_epoch_{:03d}.pth.tar'.format(cfg.n_epochs))
    torch.save(cp_dict, filename)
    print('END of training. final {} epoch for {:s}'.format(epoch, exp))
