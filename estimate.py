from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from Datasets.coco_style import COCO_STYLE
from Detectors.detector_factory import detector_factory

def estimation():
    # 创建评估数据集迭代器
    test_dataset = COCO_STYLE('./Datasets/Data', 'val')
    Detector = detector_factory['detection']
    detector = Detector('Models/model_epoch_160.pth')

    results = {}
    num_iters = len(test_dataset)
    for idx in range(num_iters):
        img_id = test_dataset.images[idx]
        img_info = test_dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(test_dataset.images_dir, img_info['file_name'])
        ret = detector.run(img_path)
        results[img_id] = ret['results']
    test_dataset.run_eval(results, "Results")

if __name__ == '__main__':
    estimation()