# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """翻转关节点（常在图片翻转后做）

    Args:
        joints (array[16, 3]): 关节点坐标
        joints_vis (array[16, 3]): 关节点可视程度
        width (int): 图片的宽度（水平翻转需要宽度）
        matched_parts (array): _description_

    Returns:
        关节点有关信息的array: joints*joints_vis, joints_vis
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        # 互换左右关节点坐标和可见程度
        #? 为什么这种方式在python里可以做到互换
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    """生成仿射变换矩阵

    Args:
        center (array[2,]): 待检测人物的中心点
        scale (float): 缩放因子
        rot (float): 旋转角度
        output_size (int): 变换后输出的大小
        shift (_type_, optional): 缩放. Defaults to np.array([0, 0], dtype=np.float32).
        inv (int, optional): 1则进行逆向转换（dst->src），0则正转换（src->dst）. Defaults to 0.

    Returns:
        array[2,3]: 仿射变换矩阵
    """
    
    # 将scale转换为array类型
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])
        
    # 这个200是官方定的
    scale_tmp = scale * 200.0 # scale*200.0 人体框的高宽为scale_tmp
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180 # 转为弧度值
    #! Q：为什么*-0.5
    #! A：就是为了取以人体框中心点为圆心,以w/2为半径的圆周上的一点
    src_dir = get_dir([0, src_w * -0.5], rot_rad) # 做旋转
    dst_dir = np.array([0,  dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    # (scale_tmp * shift)一定为正，即向右下等比移动
    src[0, :] = center + scale_tmp * shift # 点1 - 缩放后的人体框中心点坐标
    src[1, :] = center + src_dir + scale_tmp * shift # 点2 - 以人体框中心点为圆心,以w/2为半径的圆周上的一点
    #! 这里保证了仿射变换后其中心点坐标在图像中心，且保证人体框也在变换后的图像中
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5] # 点1 - 变换后的人体框中心点坐标
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir # 点2 - 以人体框中心点，偏移dst_w*-0.5

    src[2:, :] = get_3rd_point(src[0, :], src[1, :]) # 与上述两点垂直的一点
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :]) # 与上述两点垂直的一点

    # cv2.getAffineTransform(pts1,pts2) - 根据原图像的三个点的坐标，和变换后的三个点的坐标，找到对应的仿射变换矩阵
    if inv: # 是否进行逆转换
        #! 用三个点构建六个方程，解出仿射变换矩阵中的a,b,c,d,e,f六个未知数
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """对坐标点进行仿射变换

    Args:
        pt (array[2,]): 坐标点
        t (array[2,3]): 仿射变换的矩阵

    Returns:
        array[2,]: 仿射变换后的坐标点
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    """计算与a,b垂直的，特定位置的点的坐标

    Args:
        a (array[2,]): a点坐标
        b (array[2,]): b点坐标

    Returns:
        array: 特定点坐标
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """以(0,0)点为中心，进行旋转

    Args:
        src_point (float): 待旋转的坐标点
        rot_rad (float): 弧度值的旋转角度

    Returns:
        array: 旋转后的点
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    # 旋转变换公式
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img
