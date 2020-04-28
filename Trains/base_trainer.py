from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class ModelWithLoss(torch.nn.Module):
    """
    定义loss网络层
    """
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        """
        计算网络的损失
        :param batch: 数据集迭代器__getitem__的返回，使用其'input'对应的图像送入网络计算网络输出
        :return: 网络输出 + loss标量值 + loss列表
        """
        net_outputs = self.model(batch['input'])
        loss, loss_list = self.loss(net_outputs, batch)
        return net_outputs[-1], loss, loss_list

class BaseTrainer(object):
    """
    构建模型训练的基类，为了更好兼容不同的训练任务
    作用：
    （1）构建loss网络（有效分离网络模型与loss网络）
    （2）定义一个epoch训练过程
    """
    def __init__(self, model, optimizer=None):
        """
        初始化函数
        :param model: 构建的网络模型：backbone+heads，不包括loss部分，在这里构建loss
        :param optimizer: 构建的torch优化器
        """
        self.optimizer = optimizer
        # self.loss_list: loss名称列表
        # self.loss: loss网络层
        self.loss_list, self.loss = self.get_losses()
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, device):
        """
        将整个网络的参数和loss层网络的参数，放置到合适的设备上（cpu或gpus）
        :param gpus: 可以使用的gpu索引列表
        :param device: 使用的设备: 'cpu'/'cuda'
        :return: 无
        """

        # 设置loss layer的参数放置在哪里运算
        if len(gpus) > 1:
            # TODO: 将loss网络进行多GPU并行运算
            pass
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        # TODO: self.optimizer.state.values()到底是什么数据要弄明白!
        # 设置优化器状态参数放置到相应的设备中
        # self.optimizer.state是一个字典的字典，其key值自动设置
        # values是字典，因此是遍历self.optimizer.state中的字典
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, iters_per_epoch, data_loader):
        """
        模型的一个epoch训练
        :param phase: 'train' or 'val'
        :param epoch:
        :steps_per_epoch: 每个epoch迭代多少轮
        :param data_loader:
        :return:
        """
        if phase == 'train':
            self.model_with_loss.train()
        else:
            # TODO： 模型评估分支
            pass

        res_loss = 0
        num_iters = len(data_loader) if iters_per_epoch <= 0 else iters_per_epoch
        for iter_id, batch in enumerate(data_loader):
            # 判断迭代次数是否超过限制
            if iter_id >= num_iters:
                break

            # 将训练数据放入正确的计算设备中
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to('cuda', non_blocking=True)

            # 网络loss计算
            output, loss, loss_list = self.model_with_loss(batch)
            loss = loss.mean()  # loss特阵图上所有元素的loss平均值，最后得到一个标量
            if phase == 'train':
                # 优化器迭代
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            res_loss = loss_list
            print('完成迭代{}: loss值为{}'.format(iter_id, loss))
            del output, loss, loss_list
        return res_loss


    def train(self, epoch, iters_per_epoch, data_loader):
        """
        模型训练调度函数
        :param epoch:
        :param data_loader:
        :return:
        """
        return self.run_epoch('train', epoch, iters_per_epoch, data_loader)

    def save_results(self, output, batch, results):
        """
        保存模型训练的结果
        :param output:
        :param batch:
        :param results:
        :return:
        """
        raise NotImplementedError

    def get_losses(self):
        """
        子类必须实现的loss网络构造
        :return:
        """
        raise NotImplementedError