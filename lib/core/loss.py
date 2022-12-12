# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    """基于关节点的MSELoss"""
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # tensor.split(tensor, split_size_or_sections, dim=0)对张量按照指定维度dim进行分割split_size_or_sections
        # 切割后：heatmaps_pred & heatmaps_gt - {tuple : 16}(32, 1, 4096)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1) # 对num_joints维度切割，每份1个
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1) # 对num_joints维度切割，每份1个

        loss = 0
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() # 将维度上为1的维度压缩掉
            heatmap_gt = heatmaps_gt[idx].squeeze() # 将维度上为1的维度压缩掉
            # 计算热力图上每个点的得分的MSELoss（MSELoss里对Bs维度求了均值）
            # gt热力图只有可见的关节点才有热力图，否则全为0
            if self.use_target_weight: 
                # 若使用target_weight，意义在于如果是用不同的关节点权重，关节点优化力度不同
                # 若cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT存在，则关节点有着不同的权重
                loss += 0.5 * self.criterion( 
                    heatmap_pred.mul(target_weight[:, idx]), 
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else: 
                # 所有关节点同等权重优化
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
