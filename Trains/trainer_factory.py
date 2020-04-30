# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : trainer_factory.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 可创建不同的任务
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Trains.det_trainer import DetTrainer

trainer_factory = {
    'DETECTION': DetTrainer
    # TODO: 可继续增加新的训练任务, 例如实例分隔, 骨架节点检测等
}