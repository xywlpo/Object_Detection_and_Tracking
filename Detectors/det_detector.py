from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2
from .base_detector import BaseDetector
from Networks.decode import ctdet_decode
from Utils.post_process import ctdet_post_process

class DetDetector(BaseDetector):
    """
    用于目标检测的检测器
    """
    def __init__(self, model_path):
        super(DetDetector, self).__init__(model_path)

    def process(self, images):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            # dets = [[[left, top, right, bottom, score, class],[],[],[]...]], dets.size(2) = 100
            dets = ctdet_decode(hm, wh, reg=reg)
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], 20)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            x = [detection[j] for detection in detections]
            results[j] = np.concatenate(x, axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        # 返回的results是一个字典, 类别号作为key, 对应的列表[[left,top,right,bottom,score],[],[]...]作为value
        return results

    def show_results(self, image, results):
        self.class_name = ["aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                           "train", "tvmonitor"]
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > 0.1:
                    bbox = np.array(bbox, dtype=np.int32)
                    cat = int(j - 1)
                    txt = '{}{:.1f}'.format(self.class_name[cat], 1)
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
