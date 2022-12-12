# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    """计算pred坐标和target坐标之间的距离

    Args:
        preds (numpy.ndarray([batch_size, num_joints, coordinates in heatmaps])): 预测关节点在热力图上的坐标
        target (numpy.ndarray([batch_size, num_joints, coordinates in heatmaps])): GT关节点在热力图上的坐标
        normalize (_type_): PCK中的尺度因子

    Returns:
        numpy.ndarray([num_joints, batch_size]): pred坐标和target坐标之间的距离
    """
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0])) # dists - numpy.ndarray([num_joints, batch_size])
    for n in range(preds.shape[0]): # batch_size维度
        for c in range(preds.shape[1]): # num_joints
            #? 怎么是大于1呢？在读取mpii数据集的时候就已经把坐标归到(0,0)为原点了啊？
            #! 如果用>1, >1来判断关节点坐标有没有被mask掉，那么0,0点坐标没有被mask不就漏了？
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                # normalize - PCK中的尺度因子
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets) # 二范数（距离）
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """计算第i个关节点的PCK

    Args:
        dists (_type_): 第i个节点的pred和gt间的距离
        thr (float, optional): PCK的阈值T. Defaults to 0.5.

    Returns:
        _type_: 第i个关节点的PCK，无有效距离则为-1
    """
    
    # 比较的是对应元素是否相等.相等元素为false，不相等就是true
    dist_cal = np.not_equal(dists, -1) #! 前面漏检0,0坐标，这里也会导致0,0的距离不是有效距离
    num_dist_cal = dist_cal.sum() # 有效距离的个数
    if num_dist_cal > 0:
        # numpy.less(x1,x2) - 该函数用于判断x1是否小于x2
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else: # 无有效距离
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    """
    根据PCK计算accuracy（but uses ground truth heatmap rather than x,y locations）

    Args:
        output (numpy.ndarray([batch_size, num_joints, height, width])): 模型输出
        target (numpy.ndarray([batch_size, num_joints, height, width])): GT热力图
        hm_type (str, optional): 热力图的分布类型. Defaults to 'gaussian'.
        thr (float, optional): 阈值【没有用到】. Defaults to 0.5.

    Returns:
        acc: acc[0]存放所有关节点平均的PCK，其余存放第i个关节点的PCK
        avg_acc: 所有关节点平均的PCK
        cnt: 有效关节点PCK的个数
        pred: 预测的热力图关节点坐标(batch_size, num_jonts, coordinates in heatmaps)，坐标排列为(y, x)
    """
    idx = list(range(output.shape[1])) # num_joints
    norm = 1.0
    if hm_type == 'gaussian':
        # 只拿到坐标位置，而不要得分
        pred, _ = get_max_preds(output) # pred坐标为(y,x)排列
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        #! norm - PCK中的尺度因子
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10 # (batch_size, 2)
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1)) # (num_joints + 1, )
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]]) # 计算第i个关节点的PCK
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1] # 计算所有关节点的平均PCK
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc # acc[0]存放所有关节点平均的PCK，其余存放第i个关节点的PCK
    return acc, avg_acc, cnt, pred


