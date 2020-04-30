# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : det_trainer.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 根据实现不同的任务创建不同的训练网络，这里主要
# 是指相应的loss函数的构建
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_trainer import BaseTrainer
from Networks.losses import FocalLoss, RegL1Loss
import torch

class DetLoss(torch.nn.Module):
    """
    用于构建目标检测的loss层
    """
    def __init__(self):
        super(DetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()

    def forward(self, output, batch):
        """
        计算当前网络的所有loss
        :param output: 网络的预测值
        :param batch: 网络的真实标签值, 通过数据集的生成器产生
        """
        hm_loss, wh_loss, off_loss = 0, 0, 0
        output = output[0]
        output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1-1e-4)
        hm_loss += self.crit(output['hm'], batch['hm'])
        wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']
        )
        off_loss += self.crit_reg(
            output['reg'], batch['reg_mask'],
            batch['ind'], batch['reg'])

        # 整体loss计算
        loss = hm_loss + 0.1 * wh_loss + 1 * off_loss
        loss_list = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}

        # 此时再返回的loss即为整个网络的loss值, loss_list是所有loss的字典
        return loss, loss_list

class DetTrainer(BaseTrainer):
    """
    目标检测训练器
    """
    def __init__(self, model, optimizer=None):
        """
        初始化函数
        :param model: 模型网络
        :param optimizer: 优化器
        """
        super(DetTrainer, self).__init__(model, optimizer=optimizer)

    def get_losses(self):
        """
        loss层的构建放在子类中, 可以更加灵活的构建loss层
        """
        loss_list = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = DetLoss()
        return loss_list, loss