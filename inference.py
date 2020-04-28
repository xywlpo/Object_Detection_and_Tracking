from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from Detectors.detector_factory import detector_factory

def inference():
    Detector = detector_factory['detection']
    detector = Detector('Models/model_epoch_160.pth')
    ret = detector.run('Images/a2.jpg')

if __name__ == '__main__':
    inference()




