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
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    """保存标注关节点的输入图片

    Args:
        batch_image ([batch_size, channel, img_h, img_w]): 输入图片
        batch_joints ([batch_size, num_joints, 3]): 输入图像中关节点的坐标
        batch_joints_vis ([batch_size, num_joints, 1]): 关节点的可见程度（pred和gt都是用gt可见度）
        file_name (str): 保存的文件名'{}_gt.jpg'.format(prefix)或'{}_pred.jpg'.format(prefix)
        nrow (int, optional): 拼接图片的行数. Defaults to 8.
        padding (int, optional): 图片之间的间隔的padding. Defaults to 2.
    """
    
    # torchvision.utils.make_grid() - 将若干幅图像拼成一幅图像
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    # mul(255) - 恢复RGB通道的像素值；clamp(0, 255) - 限制其取值范围在[0, 255]之间
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() # 将输入图像的像素值恢复正常
    #? 为什么要浅拷贝copy()一下？
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0) # 图片个数
    xmaps = min(nrow, nmaps) # x轴个数（列数）
    ymaps = int(math.ceil(float(nmaps) / xmaps)) # y轴个数（行数）
    # 高宽加上padding才是大图上一张小图片的大小
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            # k - 第k张图片
            if k >= nmaps: # 图片处理完毕
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                # 计算在ndarr大图上的坐标位置
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]: # 可见的关节点
                    # radius=2, color=[255, 0, 0], thickness=2
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    """保存热力图图片

    Args:
        batch_image ([batch_size, channel, img_h, img_w]): 输入图像
        batch_heatmaps ([batch_size, channel, hm_h, hm_w]): 热力图(target或output)
        file_name (str): 保存的文件名'{}_hm_gt.jpg'.format(prefix)或'{}_pred_gt.jpg'.format(prefix)
        normalize (bool, optional): _description_. Defaults to True.
    """
    
    if normalize: # 最大最小标准化
        # .clone()返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        # (x-min)/(max-min)；1e-5 - 用于稳定分母
        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    # 大图的大小为grid_image，高度为batch_size*heatmap_height，宽度为(num_joints+1)*heatmap_width
    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width, # +1是因为还有一张输入图片要展示所有关节点
                           3),
                          dtype=np.uint8)

    # 获得热力图上的关节点坐标
    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        # 取batch中第i个图片，将输入图像的像素值恢复正常
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        # 取batch中第i个图片，并将其热力图的像素值恢复正常
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()
        
        # 将输入图像等比缩放到heatmap的大小
        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))
        
        # 计算当前轮应该在大图的哪个高度范围上
        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        
        # 生成当前行的每列上的图片
        for j in range(num_joints):
            # 每行第一个图片为原图上标注所有关节点的图片
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :] # 取第j个关节点的热力图
           
            # 将np.unit8格式的矩阵转化为colormap，第二个参数最常用的蓝红配色，值越小越接近蓝色，越大越接近红色
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # 给热力图上增加显示输入图片
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            # 给第j个关节点的热力图上标注所有关节点的图片
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            # 第j个关节点的热力图的宽度区域
            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            # 将第j个关节点的热力图写入大图中
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3
        # 在每行第一个位置写入原图上标注所有关节点的原始输入图片
        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    """_summary_

    Args:
        config (_type_): cfg配置文件
        input (numpy.ndarray([batch_size, num_joints, img_h, img_w])): 模型输入
        meta (dict): 当前batch的所有数据的详细内容
        target (numpy.ndarray([batch_size, num_joints, hm_h, hm_w])): Ground Truth
        joints_pred ([batch_size, num_joints, coordinates in input])): 模型预测的关节点坐标（已映射到img_h, img_w的大小）
        output ([batch_size, num_joints, coordinates in heatmaps])): 模型输出
        prefix (str): "{output_dir/train}_{i}' - 存储的图片名的前缀
    """
    if not config.DEBUG.DEBUG: # 不进行DEBUG时（DEBUG=false），不保存debug图像
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT: # 配置文件是否保存GT图片
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED: # 配置文件是否保存PRED图片
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT: # 配置文件是否保存GT的热力图
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED: # 配置文件是否保存PRED的热力图
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
