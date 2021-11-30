#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 下午7:18
# @Author  : chenxb
# @FileName: advsemiloss.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np

class CrossEntropy2d(nn.Layer):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        n, c, h, w = predict.shape
        target = target.astype('int64')
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose((0,2,1,3)).transpose((0,1,3,2))
        predict = predict[target_mask.reshape((n, h, w, 1)).tile((1, 1, 1, c))].reshape((-1, c))
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss

class BCEWithLogitsLoss2d(nn.Layer):

    def __init__(self, reduction='mean', ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, reduction=self.reduction)
        return loss