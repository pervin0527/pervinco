#-*- coding:utf-8 -*-

"""
Maximum Entropy Classification Loss
"""
import torch
from torch import nn
import torch.nn.functional as F

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

        b = b.sum(1)
        b = b.mean()

        return b


class MecNetLoss(nn.Module):
    def __init__(self, learn_alpha=False, num_cls=1000, num_ang=72, num_flr=3, alphap=0.0, alphaa=0.0, alphaf=0.0):
        super(MecNetLoss, self).__init__()
        self.alphap = nn.Parameter(torch.Tensor([alphap]), requires_grad=learn_alpha)
        self.alphaa = nn.Parameter(torch.Tensor([alphaa]), requires_grad=learn_alpha)
        self.alphaf = nn.Parameter(torch.Tensor([alphaf]), requires_grad=learn_alpha)

        self.num_cls = num_cls
        self.num_ang = num_ang
        self.num_flr = num_flr

    def forward(self, pred, targ):
        pred_pos, pred_cls, pred_ang, pred_flr, pred_mcls = torch.split(pred, (6,self.num_cls,self.num_ang,self.num_flr,self.num_cls), dim=2)
        targ_pos, targ_cls, targ_ang, targ_flr, targ_mcls = torch.split(targ, (6,self.num_cls,1,1,1), dim=2)

        # classification loss for position using soft labels
        self.c_loss_fn = nn.KLDivLoss()
        s = pred_cls.size()
        pred_cls_soft = torch.softmax(pred_cls.view(-1, *s[2:]), dim=1)
        cls_loss = torch.exp(-self.alphap) * self.c_loss_fn(torch.log(pred_cls_soft), targ_cls.view(-1, *s[2:]))+ self.alphap

        # large margin softmax classification loss for position
        self.m_loss_fn = nn.CrossEntropyLoss()
        s = pred_mcls.size()
        mcls_loss = torch.exp(-self.alphaf) * self.m_loss_fn(pred_mcls.view(-1, *s[2:]), targ_mcls.view(-1).long()) + self.alphaf

        self.e_loss_fn = HLoss()
        ent_loss = torch.exp(-self.alphaf) * self.e_loss_fn(pred_mcls.view(-1, *s[2:]))+ self.alphaf

        loss =  cls_loss + mcls_loss + ent_loss

        return loss


class MecNetANGLoss(nn.Module):
    def __init__(self, learn_alpha=False, num_cls=1000, num_ang=72, num_flr=3, \
                alphap=0.0, alphaa=0.0, alphaf=0.0):

        super(MecNetANGLoss, self).__init__()
        self.alphap = nn.Parameter(torch.Tensor([alphap]), requires_grad=learn_alpha)
        self.alphaa = nn.Parameter(torch.Tensor([alphaa]), requires_grad=learn_alpha)
        self.alphaf = nn.Parameter(torch.Tensor([alphaf]), requires_grad=learn_alpha)


        self.num_cls = num_cls
        self.num_ang = num_ang
        self.num_flr = num_flr

    def forward(self, pred, targ):

        dim = pred.size(2)
        pred_pos, pred_cls, pred_ang, pred_flr, pred_mcls, pred_mangcls = torch.split(pred, (6,self.num_cls,self.num_ang,self.num_flr,self.num_cls,self.num_ang), dim=2)
        targ_pos, targ_cls, targ_ang, targ_flr, targ_mcls, targ_mangcls = torch.split(targ, (6,self.num_cls,self.num_ang,1,1,1), dim=2)


        # classification loss for position using soft labels
        self.c_loss_fn = nn.KLDivLoss()
        s = pred_cls.size()
        pred_cls_soft = torch.softmax(pred_cls.view(-1, *s[2:]), dim=1)

        cls_loss = torch.exp(-self.alphap) * self.c_loss_fn(torch.log(pred_cls_soft), targ_cls.view(-1, *s[2:]))+ self.alphap

        # large margin softmax classification loss for position
        self.m_loss_fn = nn.CrossEntropyLoss()
        s = pred_mcls.size()
        mcls_loss = torch.exp(-self.alphaa) * self.m_loss_fn(pred_mcls.view(-1, *s[2:]), targ_mcls.view(-1).long()) + self.alphaa

        self.e_loss_fn = HLoss()
        ent_loss = torch.exp(-self.alphaa) * self.e_loss_fn(pred_mcls.view(-1, *s[2:]))+ self.alphaa


        # classification loss for angle using soft labels
        self.a_loss_fn = nn.KLDivLoss()
        s = pred_ang.size()
        pred_ang_soft = torch.softmax(pred_ang.view(-1, *s[2:]), dim=1)

        angcls_loss = torch.exp(-self.alphap) * self.a_loss_fn(torch.log(pred_ang_soft), targ_ang.view(-1, *s[2:]))+ self.alphap

        # large margin softmax classification loss for position
        self.ma_loss_fn = nn.CrossEntropyLoss()
        s = pred_mangcls.size()
        mangcls_loss = torch.exp(-self.alphaa) * self.ma_loss_fn(pred_mangcls.view(-1, *s[2:]), targ_mangcls.view(-1).long()) + self.alphaa

        self.ea_loss_fn = HLoss()
        angent_loss = torch.exp(-self.alphaa) * self.ea_loss_fn(pred_mangcls.view(-1, *s[2:]))+ self.alphaa



        # total loss
        loss =  cls_loss + mcls_loss + ent_loss +\
                angcls_loss + mangcls_loss + angent_loss

        return loss
