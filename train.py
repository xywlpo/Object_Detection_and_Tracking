# -*- coding: utf-8 -*-
# @Time : 2020年4月30日
# @Author : Jiang Nan
# @File : train.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 模型训练的主函数
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Datasets.coco_style import COCO_STYLE
from Networks.model import create_model, save_model
from Trains.trainer_factory import trainer_factory
from config import Config

def main():
    """
    模型训练函数
    :return: none
    """

    # 创建Tensorboard对象
    writer = SummaryWriter('Log')

    # 创建参数配置类
    config = Config()
    config.CURRENT_PROCESS = 'TRAIN'

    # 创建训练数据集迭代器
    train_dataset = COCO_STYLE(config, 'train')

    # 创建模型网络结构
    model = create_model(config)
    print('model has been created')

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)

    # 创建训练器
    device = torch.device(config.DEVICE)
    Trainer = trainer_factory[config.TASK]
    trainer = Trainer(model, optimizer)

    # 将包括loss在内的整个网络, 以及优化器都放到合适的计算设备上
    trainer.set_device(config.GPU_LIST, device)

    # 创建数据生成器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUMBERS_OF_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # 开始训练
    # TODO: 尚未加入验证集的评估
    print('starting training...')
    for epoch in range(config.ALL_EPOCHES):
        loss_list = trainer.train(epoch, config.STEPS_OF_EPOCH, train_loader)
        for key in loss_list:
            writer.add_scalar("train_{}".format(key), loss_list[key], epoch)
        if epoch % config.INTERVAL_OF_EPOCH == 0:
            save_model('./model_epoch_{}.pth'.format(epoch), epoch, model, optimizer)

if __name__ == '__main__':
    main()


