from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Config(object):
    """
    参数配置类
    """

    #########################################################
    ######################## 整体参数 #########################
    #########################################################

    # 当前是训练（TRAIN）还是推理（INFERENCE）
    CURRENT_PROCESS = 'TRAIN'

    # 当前要执行的任务, 例如目标检测/实例分割/骨架节点检测等
    TASK = 'DETECTION'

    # 运行设备
    DEVICE = 'cuda'

    # 可使用的GPUs列表
    GPU_LIST = [0]

    ### ================= 模型推理整体参数 ====================

    # 存放模型文件的路径
    INFERENCE_MODEL_PATH = ''

    # 推理时每张图像最多检测目标个数
    MAX_PER_IMAGE = 100

    # 推理阈值
    INFERENCE_THRESHOLD = 0.1

    ### ================= 模型训练整体参数 ====================

    # 是否使用离线的预训练模型
    OFFLINE_PRETRINED_MODEL = True

    # 优化器的学习率
    LEARNING_RATE = 5e-4

    # batchsize
    BATCH_SIZE = 2

    # 工作的线程数量
    NUMBERS_OF_WORKERS = 1

    # 训练时迭代的轮次
    ALL_EPOCHES = 50

    # 每轮迭代多少次
    STEPS_OF_EPOCH = 80

    # 多少轮次保存一个模型数据
    INTERVAL_OF_EPOCH = 1


    #########################################################
    ###################### 数据库相关参数 ######################
    #########################################################

    # 数据集路径
    DATA_DIR = "Datasets/Data/Kaggle/"

    # 注释文件字典
    ANN_NAME = {'train':'trainval2007', 'val':'test2007'}

    # 类别名称
    CLASS_NAME = ["corn"]

    # 类别数量, 不包含背景类
    NUM_CLASSES = 1

    # 数据集默认分辨率, 输入图像统一缩放的尺寸, 顺序为 [height, width]
    RESOLUTION = [1024, 1024]

    # 数据集的各通道均值, 用于对输入图像进行0中心化与归一化
    MEAN = [0.485, 0.456, 0.406]

    # 数据集的各通道标准差, 用于对输入图像进行0中心化与归一化
    STD = [0.229, 0.224, 0.225]

    # 限制一张图片上最多的物体数量
    MAX_OBJECTS = 100


    #########################################################
    ##################### 网络模型相关参数 #####################
    #########################################################

    # 选择的BACKBONE网络结构
    ARCH = 'resnet_50'

    # 选择的网络头heads
    HEADS = {'hm': NUM_CLASSES, 'wh': 2, 'reg':2}

    ### 网络结构01 ========= RESNET网络参数 ====================

    # ResNet预训练模型链接
    RESNET_MODEL_URLS = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    # BN参数
    RESNET_BN_MOMENTUM = 0.1

