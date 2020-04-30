# -*- coding: utf-8 -*-
# @Time : 2020年4月30日
# @Author : Jiang Nan
# @File : coco_style.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 用于生成数据的迭代器
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import math
import json
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import torch.utils.data as Data
from Utils.image import gaussian_radius, draw_umich_gaussian
from Utils.image import get_affine_transform, affine_transform

class COCO_STYLE(Data.Dataset):
    """
    COCO格式数据集
    继承自torch.utils.data.Dataset
    """

    def __init__(self, config, stage):
        """
        参数初始化
        :param config: 参数配置对象
        :param stage: 指定当前属于 train or inference or estimate
        :return: None
        """
        super(COCO_STYLE, self).__init__()
        self.dataset_dir = config.DATA_DIR
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.ann_path = os.path.join(self.dataset_dir, 'annotations', '{}.json').format(config.ANN_NAME[stage])
        self.class_name = config.CLASS_NAME
        self.num_classes = config.NUM_CLASSES
        self.category_ids = [i for i in range(self.num_classes)]
        self.resolution = config.RESOLUTION
        self.mean = np.array(config.MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(config.STD, dtype=np.float32).reshape(1, 1, 3)
        self.max_objs = config.MAX_OBJECTS
        self.coco = coco.COCO(self.ann_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

    def __len__(self):
        """
        构建计算样本数量的函数
        :return: 样本数量
        """
        return self.num_samples

    def ltwhBOX_to_ltrbBOX(self, box):
        """
        bbox坐标形式转换
        left, top, width, height -> left, top, right, bottom
        :param box: box[left, top, width, height] -> coco默认的bbox格式
        :return: bbox[left, top, right, bottom]
        """
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def __getitem__(self, item):
        """
        返回一个样本及其标注数据
        TODO：增加数据增强操作
        :param item: 样本序号
        :return: 样本及其标注信息
        """
        # 获取图像路径, 标注内容, 一张图像上允许的目标数量
        img_id = self.images[item]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.images_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        # 按照原始图像最长边将图像缩放到统一尺寸(resolution)，采用仿射变换实现
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1]/2., img.shape[0]/2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1])*1.0
        input_h, input_w = self.resolution[0], self.resolution[1]
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        # 图像归一化 + 零均值化
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # 为了与pytorch的tensor对应 [batchsize, channels, h, w]
        img = img.transpose(2, 0, 1)

        # 最后的feature map尺寸和变换矩阵
        output_h = input_h // 4
        output_w = input_w // 4
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # 存储目标中心位置的热图heat_map, 中心点位置的偏移值center_offset, 目标的width和height
        heat_map = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        center_offset = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh_size = np.zeros((self.max_objs, 2), dtype=np.float32)

        # 用于计算loss的辅助参数
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # 遍历当前图像中的每个目标
        for k in range(num_objs):
            # 获取bbox和class_id
            ann_info = anns[k]
            bbox = self.ltwhBOX_to_ltrbBOX(ann_info['bbox'])
            class_id = int(self.category_ids[ann_info['category_id'] - 1])

            # bbox变换到最后输出的feature map上，之后的计算都在feature map上
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            # 对变换后的bbox坐标位置进行限制, 防止超界
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                center_point = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], dtype=np.float32)
                center_point_int = center_point.astype(np.int32)
                draw_umich_gaussian(heat_map[class_id], center_point_int, radius)
                wh_size[k] = 1.*w, 1.*h
                ind[k] = center_point_int[1] * output_w + center_point_int[0]
                center_offset[k] = center_point - center_point_int
                reg_mask[k] = 1
        ret = {'input': img, 'hm': heat_map, 'reg': center_offset, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh_size}
        return ret

    def to_float(self, x):
        """
        将x转换为保留2位小数的float型
        :param x: 输入数据
        :return: 返回保留2位小数的float型数据
        """
        return float("{:.2f}".format(x))

    def convert_eval_format(self, res):
        """
        将推理识别的结果转换为coco数据集标注文件的相应格式
        :param res: 评估数据集的推理结果
        :return: 与coco数据集标注文件相同的评估结果格式
        """
        detections = []
        for image_id in res:
            for cls_ind in res[image_id]:
                for bbox in res[image_id][cls_ind]:
                    # bbox需转换为coco的bbox格式：left, top, width, height
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self.to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(self.category_ids[cls_ind]),
                        "bbox": bbox_out,
                        "score": self.to_float(score)
                    }
                    detections.append(detection)
        return detections

    def run_eval(self, results, save_dir):
        """
        基于推理后的结果文件, 进行AP的计算
        :param results: 评估数据集中每张图像的识别结果
        :param save_dir: 评估结果文件保存路径
        :return: None
        """
        # 将识别的结果转换成coco评估的格式，并保存为json文件
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()















