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
    def __init__(self, model_path):
        arch = 'resnet_18'
        heads = {'hm': 20, 'wh': 2, 'reg': 2}
        self.model = create_model(arch, heads)
        self.model = load_model(self.model, model_path)
        self.model = self.model.to('cuda')
        self.model.eval()   # model.eval()必须要有，这样才能保证dropout和BN层使用推理模式
        self.mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = 20

    def pre_process(self, image, meta=None):
        height, width = image.shape[0:2]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        inp_height, inp_width = 384, 384
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

    def run(self, image_or_path, meta=None):

        # 读取原始图像
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            image = cv2.imread(image_or_path)

        images, meta = self.pre_process(image, meta)
        images = images.to('cuda')
        output, dets = self.process(images)
        dets = self.post_process(dets, meta)
        detections = []
        detections.append(dets)
        results = self.merge_outputs(detections)
        self.show_results(image, results)
        return {'results': results}