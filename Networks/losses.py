from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from Utils.utils import _transpose_and_gather_feat

def neg_loss(pred, gt):
    """
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    :param pred: (batch x c x h x w)
    :param gt: (batch x c x h x w)
    :return: loss标量数值
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    pos_loss = torch.log(pred)*torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * torch.pow(1 - gt, 4) * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -1 * neg_loss
    else:
        loss = -1 * (pos_loss + neg_loss) / num_pos
    return loss

class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FocalLoss(torch.nn.Module):
    """
    定义Focal Loss，用于中心点的损失函数
    """
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = neg_loss

    def forward(self, pred, gt):
        return self.neg_loss(pred, gt)



