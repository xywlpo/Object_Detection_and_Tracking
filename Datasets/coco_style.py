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
    (1)当前采用Pascal VOC 2007的数据，并将其转换为COCO格式
    (2)当前类的__getitem__适用于centernet的标注格式
    """

    def __init__(self, data_dir, stage):
        """
        初始化基本参数
        :param data_dir: 数据库总目录
        :param stage: train / val / test
        """
        super(COCO_STYLE, self).__init__()
        self.dataset_dir = os.path.join(data_dir, 'VOC')
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        ann_name = {'train':'trainval2007', 'val':'test2007'}
        self.annot_path = os.path.join(self.dataset_dir, 'annotations', 'pascal_{}.json').format(ann_name[stage])

        # Pascal VOC 的类别id与self.class_name的类别索引号对应
        self.class_name = ['__background__', "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                           "train", "tvmonitor"]

        # 将类别索引开始序号从1映射为0，便于后续通过类别序号索引heat map方便
        self.valid_ids = np.arange(1, 21, dtype=np.int32)
        self.category_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.max_objs = 50
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.num_classes = 20
        self.default_resolution = [384, 384]
        self.mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        """
        构建计算样本数量的函数
        :return: 样本数量
        """
        return self.num_samples

    def coco_box_to_bbox(self, box):
        """
        将coco格式的bbox格式转换为常用的格式
        :param box: box格式 top,left,bottom,right
        :return:bbox格式 top,left,width,height
        """
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __getitem__(self, item):
        """
        返回一个样本及其标注数据，尚未增加数据增强操作
        :param item: 样本序号
        :return: 样本及其标注信息
        """
        img_id = self.images[item]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.images_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        # 按照原始图像最长边将图像缩放到统一尺寸(default_resolution)，采用仿射变换实现
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1]/2., img.shape[0]/2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1])*1.0
        input_h, input_w = self.default_resolution[0], self.default_resolution[1]
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        # 图像归一化 + 零均值化
        # 由于上面对图像缩放后会有黑边，此时会出现负数，为了避免对模型训练的干扰，对<0的部分置0
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)    # 为了与pytorch的tensor对应 [batchsize, channels, h, w]

        # 最后的feature map尺寸，以及最后feature map的变换
        output_h = input_h // 4
        output_w = input_w // 4
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        heat_map = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        center_offset = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh_size = np.zeros((self.max_objs, 2), dtype=np.float32)

        # 用于计算loss的辅助参数
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # 遍历当前图像中的每个目标
        # gt_det = []
        for k in range(num_objs):
            ann_info = anns[k]
            bbox = self.coco_box_to_bbox(ann_info['bbox'])
            class_id = int(self.category_ids[ann_info['category_id']])

            # bbox变换到最后输出的feature map上，之后的计算都在feature map上
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
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
                center_offset[k] = center_point - center_point_int      # 中心点坐标的偏差量
                reg_mask[k] = 1
                # gt_det.append([center_point[0] - w / 2, center_point[1] - h / 2, center_point[0] + w / 2, center_point[1] + h / 2, 1, class_id])
        ret = {'input': img, 'hm': heat_map, 'reg': center_offset, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh_size}
        return ret

    def to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, res):
        """
        转换识别结果为coco评估的格式
        :param res:
        :return:
        """
        detections = []
        for image_id in res:
            for cls_ind in res[image_id]:
                category_id = self.valid_ids[cls_ind - 1]
                for bbox in res[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self.to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    detections.append(detection)
        return detections

    def run_eval(self, results, save_dir):
        """
        运行评估
        :param results: 评估数据集中每张图像的识别结果
        :param save_dir: 评估结果保存路径
        :return:
        """
        # 将识别的结果转换成coco评估的格式，并保存为json文件
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()















