# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : base_detector.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 检测器基类, 可继承其构建解决不同任务的检测器
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
from Networks.model import create_model, load_model
from Utils.image import get_affine_transform

class BaseDetector(object):
    """
    检测器的基类
    """
    def __init__(self, config):

        # 创建网络
        self.model = create_model(config)

        # 加载模型文件
        self.model = load_model(self.model, config.INFERENCE_MODEL_PATH)

        # 将网络放置到合适的设备上
        self.device = config.DEVICE
        self.model = self.model.to(self.device)

        # model.eval()必须要有，这样才能保证dropout和BN层使用推理模式
        self.model.eval()

        # 基本参数配置
        self.mean = np.array(config.MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(config.STD, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = config.MAX_PER_IMAGE
        self.num_classes = config.NUM_CLASSES
        self.class_name = config.CLASS_NAME
        self.resolution = config.RESOLUTION

    def pre_process(self, image, meta=None):
        """
        图像预处理函数:
        1. 将图像缩放到训练时的大小
        2. 进行归一化与中心化
        3. 转换成torch.tensor类型
        :param image: numpy格式的图像
        :return: 返回预处理后的图像, 以及参数信息meta
        """
        height, width = image.shape[0:2]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        inp_height, inp_width = self.resolution[0], self.resolution[1]
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)      #归一化图像
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 'out_height': inp_height // 4, 'out_width': inp_width // 4}
        return images, meta

    def process(self, images):
        raise NotImplementedError

    def post_process(self, dets, meta):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def show_results(self, image, results):
        raise NotImplementedError

    def run(self, path, meta=None):
        """
        模型推理的主要函数
        :param path: 图片路径
        """

        # 读取原始图像
        image = cv2.imread(path)

        # 对图像进行预处理
        images, meta = self.pre_process(image, meta)
        images = images.to(self.device)

        # 模型推理
        # 返回的dets中bbox是基于输出的特征图上的, 因此还需
        # 变换到原始图像上的坐标位置
        output, dets = self.process(images)

        # 后处理: 将原始输出特征图上的坐标位置, 还原到原始图像上, 并改变检测结果的组织形式
        dets = self.post_process(dets, meta)

        # 若检测的目标个数大于预设的最大值, 则将较小的多出数量的目标删除
        detections = []
        detections.append(dets)
        results = self.merge_outputs(detections)

        # 绘制显示结果
        self.show_results(image, results)
        return {'results': results}