# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps

    Args:
        batch_heatmaps (numpy.ndarray([batch_size, num_joints, height, width])): 带批量维度的热力图

    Returns:
        (numpy.ndarray([batch_size, num_joints, coordinates in heatmaps]), 
            numpy.ndarray([batch_size, num_joints, 1])): preds(y,x排列的坐标), maxvals，预测的坐标点和坐标点的热力图得分
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2) # 获得在第2维度上，最大的值的下标
    maxvals = np.amax(heatmaps_reshaped, 2) # 获得在第2维度上，最大的值

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # np.tile(idx, (1, 1, 2)) - 将idx沿着0和1轴扩大1倍，沿着2轴扩大2倍。如果扩大倍数只有一个，默认为0轴
    #* 目的在于：preds中要存放最大值在热力图中的x,y坐标
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width # 求y轴坐标
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) # 求x轴坐标（向下取整）

    # np.greater(x, a)，逐个判断a中元素是否大于0.0；并扩增
    #? 这里0.0是不是可以换成人为设定的阈值，从而筛除不满足要求的低质量预测点
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32) # 转为0或1

    #! 有问题，如果坐标在0,0点的val小于0.0，那么mask之后还是0,0点
    preds *= pred_mask # mask掉热力图得分较低的关节点坐标
    # 并没有对热力图得分进行mask，只对坐标进行了mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
