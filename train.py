from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Datasets.coco_style import COCO_STYLE
from Networks.model import create_model, save_model
from Trains.trainer_factory import trainer_factory
from collections import defaultdict

def main():
    """
    模型训练
    :return: none
    """
    # 创建Tensorboard
    writer = SummaryWriter('Log')

    # 创建训练数据集迭代器
    train_dataset = COCO_STYLE('./Datasets/Data', 'train')

    # 创建模型网络结构
    arch = 'resnet_18'
    heads = {'hm': 20, 'wh': 2, 'reg': 2}
    model = create_model(arch, heads)
    print('model has been created')

    # 创建优化器
    learning_rate = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # 创建训练器
    device = torch.device('cuda')
    Trainer = trainer_factory['detection']
    trainer = Trainer(model, optimizer)
    trainer.set_device([0], device)

    # 创建数据生成器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 开始训练
    print('starting training...')
    for epoch in range(200):
        loss_list = trainer.train(epoch, 10, train_loader)
        for key in loss_list:
            writer.add_scalar("train_{}".format(key), loss_list[key], epoch)
        if epoch % 40 == 0:
            save_model('./model_epoch_{}.pth'.format(epoch), epoch, model, optimizer)

if __name__ == '__main__':
    main()


