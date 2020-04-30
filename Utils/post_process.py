# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : post_process.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 后处理函数
# -*- 功能说明 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Utils.image import transform_preds
import numpy as np

def ctdet_post_process(dets, c, s, h, w, num_classes):
  """
  将原始输出特征图上的坐标位置, 还原到原始图像上, 并改变检测结果的组织形式
  """
  ret = []

  # dets.shape[0]对应的是batchsize这个维度
  for i in range(dets.shape[0]):

    # 分别将bbox的(left, top)和(right, bottom)变换到原始图像上
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))

    # 获取当前目标的类别
    classes = dets[i, :, -1]

    # 重新组织检测结果
    # 将原来的列表 [[left, top, right, bottom, score, class],[]...]
    # 转换成字典 top_preds = {ClassID:[[left, top, right, bottom, score],[...],...],...}
    top_preds = {}
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)

  # 最后返回的形式如下
  # [{ClassID:[[left, top, right, bottom, score],[...],...],...}]
  return ret