# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : inference.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 推理主函数
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from config import Config
from Detectors.detector_factory import detector_factory

def inference():

    # 创建参数配置对象
    config = Config()
    config.CURRENT_PROCESS = 'INFERENCE'
    config.INFERENCE_MODEL_PATH = 'Models/model_epoch_160.pth'

    # 构建目标检测器
    Detector = detector_factory[config.TASK]
    detector = Detector(config)

    # 运行模型检测器
    ret = detector.run('Images/a5.jpg')

if __name__ == '__main__':
    inference()




