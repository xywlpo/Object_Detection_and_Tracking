## Introduction
This project aims to build up a general platform of object detecting and tracking. It will contain most popular algorithms, such as anchor-based and anchor-free object detecting algorithms. For helping more people understand and be easy to use, these code will be annotated in detail. Since we want to apply computer vision to practical application, the c++ code for implementation deployment will be provided.
## Current Process
The platform currently supports CenterNet algorithm, which is a anchor-free object detection algorithm. For the future we will support more types of algorithm. 

- [x] support COCO dataset style
- [x] support ResNet as BackBone
- [ ] support RegNet as BackBone
- [x] support model training based on pre-trained model
- [x] support model test and model estimation
## Installation
1. The code was tested on Ubuntu 18.04, with Anaconda Python 3.7, Pytorch 1.5.0 and Torchvision 0.5
2. Install COCOAPI
`git clone https://github.com/cocodataset/cocoapi.git $COCOAPI`
`cd $COCOAPI/PythonAPI`
`make`
`python setup.py install --user`
