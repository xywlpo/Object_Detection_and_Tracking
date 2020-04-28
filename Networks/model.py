from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .NetArch.resnet import get_center_resnet

# 通过model_factory字典可以灵活的切换不同的基础网络结构
model_factory={
    'resnet':get_center_resnet
}

def create_model(arch, heads):
    """
    创建网络结构
    :param arch: 基础网络结构
    :param heads: dict类型，存储网络头的名字和相应的feature map channel
    :return: pytorch网络模型
    """
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads)
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    """
    加载模型进行推理 or 加载模型恢复训练
    TODO: 应该区分训练与推理，主要是在推理时参数必须严格满足建立的模型
    """

    # 读取模型文件，并将其放置到GPU上
    # torch.load会先在CPU上进行反序列化，然后将数据推送到map_location指定的设备上
    # 返回的checkpoint是一个长度为3的字典，其关键字包括：
    # (1)'epoch': 当前模型文件是迭代了多少轮次
    # (2)'state_dict': 保存所有网络层的参数
    # (3)'optimizer': 保存优化器的参数
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    # 获取模型网络层的参数值
    state_dict = checkpoint['state_dict']   # 返回的state_dict是一个有序字典
    model_state_dict = model.state_dict()   # torch.nn.Module.state_dict()返回模型网络层的参数bias, weight

    # 检查网络参数是否对应，若存在不对应情况需做下述处理
    # （1）如加载的模型文件pth与代码创建的模型model，存在相同的key，但是对应的shape不同，
    #     则将代码创建的key对应的随机参数覆盖模型文件的key的参数
    # （2）若pth中的key不在model中，则将其舍去。即加载模型时strict=false
    # （3）若model中的key不在pth的key中，则在pth中创建相应的key
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # 若resume训练过程
    start_epoch = 0
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model