# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : det_detector.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 目标检测类
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2
from .base_detector import BaseDetector
from Networks.decode import ctdet_decode
from Utils.utils import ctdet_post_process

class DetDetector(BaseDetector):
    """
    用于目标检测的检测器
    """
    def __init__(self, config):
        super(DetDetector, self).__init__(config)
        self.threshold = config.INFERENCE_THRESHOLD

    def process(self, images):
        """
        模型推理函数
        :param image: 经过预处理后的图像
        :return:
        1. output: 模型原始的输出结果
        2. dets: 返回最多100个目标的检测结果
                 !!!!! 注意其bouding box是基于输出特征图
                 的坐标位置, 因此还需还原到原始图像上 !!!!
        """
        with torch.no_grad():

            # 模型前向传播过程
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']

            # 返回下列格式的检测结果, 返回最多100个目标
            # dets = [[[left, top, right, bottom, score, class],[],[],[]...]], dets.size(2) = 100
            dets = ctdet_decode(hm, wh, reg=reg)
            return output, dets

    def post_process(self, dets, meta, scale=1):
        """
        后处理函数
        1. 将检测结果放到cpu上, 并转换成numpy格式
        2.
        """
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])

        # 以下返回的dets = [{ClassID:[[left, top, right, bottom, score],[...],...],...}]
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(self.num_classes):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale

        # 最后返回 dets[0] = {ClassID:[[left, top, right, bottom, score],[...],...],...}
        return dets[0]

    def merge_outputs(self, detections):
        """
        根据目标检测的score对结果进行排序, 并删除超出MAX_PER_IMAGE的部分
        """

        # results得到一个字典, key是类别id, value是np数组, 其每个元素为
        # [left, top, right, bottom, score]
        results = {}
        for j in range(self.num_classes):
            x = [detection[j] for detection in detections]
            results[j] = np.concatenate(x, axis=0).astype(np.float32)

        # 按照水平方向将数组的score都堆叠起来, 即形成一个100个score的列表
        scores = np.hstack(
            [results[j][:, 4] for j in range(self.num_classes)])

        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(self.num_classes):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        # 返回的results是一个字典, 类别号作为key, 对应的列表[[left,top,right,bottom,score],[],[]...]作为value
        return results

    def show_results(self, image, results):
        """
        用于显示检测的结果
        """
        for j in range(self.num_classes):
            for bbox in results[j]:
                if bbox[4] > self.threshold:
                    bbox = np.array(bbox, dtype=np.int32)
                    cls_id = int(j)
                    print(cls_id)
                    txt = self.class_name[cls_id]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1] - cat_size[1] - 2),
                                  (bbox[0] + cat_size[0], bbox[1] - 2), (255, 0, 0), -1)
                    cv2.putText(image, txt, (bbox[0], bbox[1] - 2),
                                font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow('results', image)
        cv2.waitKey(0)
