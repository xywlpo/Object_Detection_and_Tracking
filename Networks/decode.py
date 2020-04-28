from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from Utils.utils import _gather_feat, _transpose_and_gather_feat

def _nms(heat, kernel=3):
    """
    (1)使用3x3卷积，每个点处放置其3x3邻域内的最大值
    (2)找到最大值的位置设置为1.0，其余位置设置为0
    (3)heat * keep 对应点相乘，恢复最大值点的真实数值，其他位置为0
    :param heat:
    :param kernel:
    :return:
    """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    # 分别获取每个样本对应的heatmap中每个channel的最大的k个数值
    batch, cat, height, width = scores.size()
    # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
    # 沿着input的最后一维，返回最大的k个元素值与索引
    # topk_scores = [batchsize, channels, 100个socres]
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    # 获取k个最大值的heatmap坐标(x,y),即不考虑是在哪个类别的channel上，仅考虑坐标位置
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # 区分topk_inds和topk_ind的不同：
    # topk_inds：每个类别得分前100的heatmap的索引位置（考虑所有类别的heatmap）
    # topk_ind：基于topk_scores所有类别得分前100的topk_scores的位置
    # 因此下面需要使用topk_ind来索引topk_inds中存放的heatmap上的坐标位置
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    # 返回所有类别前100得分的得分值、heatmap上的整体索引、类别、heatmapx\y坐标索引（不区分是哪张feature map上）
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    # 返回的detections是[[[left, top, right, bottom, score, class],[],[],[]...]]
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections
