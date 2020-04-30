# -*- coding: utf-8 -*-
# @Time : 2020年4月20日
# @Author : Jiang Nan
# @File : detector_factory.py
# @Software: PyCharm
# @contact: xywlpo@163.com
# -*- 功能说明 -*-
# 根据不同任务调度检测器
# -*- 功能说明 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Detectors.det_detector import DetDetector

detector_factory = {
  'DETECTION': DetDetector,
}