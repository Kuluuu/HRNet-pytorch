# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200 # 像素标准化参数
        self.flip_pairs = [] # 成对的关节点【在子类中赋值】
        self.parent_ids = [] # 组建关节点的骨架关系【在子类中赋值】

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR # 输出路径
        self.data_format = cfg.DATASET.DATA_FORMAT # 数据格式

        self.scale_factor = cfg.DATASET.SCALE_FACTOR # 缩放因子
        self.rotation_factor = cfg.DATASET.ROT_FACTOR # 旋转角度
        self.flip = cfg.DATASET.FLIP # 反转
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY # 半身关节点个数
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY # 采取半身增强的概率
        self.color_rgb = cfg.DATASET.COLOR_RGB # 是否转为RGB格式

        self.target_type = cfg.MODEL.TARGET_TYPE # 热力图的分布类型
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE) # 输入图像大小
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE) # 热力图大小
        self.sigma = cfg.MODEL.SIGMA # 标注偏差
        
        # 【似乎没用到】
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT 
        self.joints_weight = 1

        self.transform = transform # 外部定义的transform操作
        self.db = [] # 数据集【在子类中赋值】

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        """_summary_

        Args:
            joints (_type_): 单个人体的16个关节点的坐标，但是3d
            joints_vis (_type_):  单个人体的16个关节点的可见程度，但是3d

        Returns:
            _type_: _description_
        """
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):        
        """
            深复制，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
            与等号赋值不同，等号复制类似于贴标签，两者实质上是同一段内存。
            像列表这样的变量，可以用深复制复刻，从而建立一个完全的新变量，如果用等号给列表赋值，则新变量的改变将会引起原变量的随之改变。
        """
        db_rec = copy.deepcopy(self.db[idx]) # 避免改变原数据内容

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip': # 若数据的格式为zip
            from utils import zipreader
            # cv2.IMREAD_COLOR - 默认使用该种标识。加载一张彩色图片，忽视它的透明度。
            # cv2.IMREAD_IGNORE_ORIENTATION - 忽略EXIF中的方向标识，不旋转图像
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb: # 转为RGB格式
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None: # 没能读取到数据
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d'] # 单个人体的16个关节点坐标，但是3d
        joints_vis = db_rec['joints_3d_vis'] # 单个人体的16个关节点的可见程度，但是3d

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train: # 训练过程
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body # 存在一半以上的关节点是可见的
                and np.random.rand() < self.prob_half_body): # 以prob_half_body概率进行half_body_transform的数据增强
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor # 缩放因子
            rf = self.rotation_factor # 旋转因子
            # np.clip做最大值最小值截断，numpy.clip(a, a_min, a_max, out=None)
            # 随机缩放（缩放比例随机，[1 - sf, 1 + sf]）
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            # 随机旋转（角度随机，[-rf*2, rf*2]）
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0 # 以0.6的概率进行旋转增强

            if self.flip and random.random() <= 0.5: # 以0.5的概率进行翻转
                data_numpy = data_numpy[:, ::-1, :] # [::-1] 顺序相反操作，第二维的顺序反向
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1 # 中心点的x坐标也要做变换

        trans = get_affine_transform(c, s, r, self.image_size) # 仿射变换矩阵
        #! Q：为什么直接对每个点做类似get_affine_transform的操作？
        #! A：因为获得仿射变换矩阵后，用矩阵进行变换更快
        #? 怎么保证做完仿射变换之后，待检测的人仍在input图像中，且未遭到切割？
        #? 做完仿射变换，c和s怎么还是原图的大小？
        input = cv2.warpAffine( # 对原始图像进行仿射变换
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])), # 输出图像的大小
            flags=cv2.INTER_LINEAR) # flags - 插值方法（线性插值）

        
        cv2.imwrite('input.png', input)
        cv2.imwrite('data_numpy.png', data_numpy)

        if self.transform: # 再对数据进行外部自定义的transform
            input = self.transform(input)

        # 对关节点坐标也进行相同的仿射变换
        for i in range(self.num_joints): 
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target) # 热力图转为张量
        target_weight = torch.from_numpy(target_weight) # 节点可视程度转为张量

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        """挑选一些标注较准的数据？【没有用到】

        Args:
            db (_type_): 全部数据集

        Returns:
            _type_: 经过挑选过的数据集
        """
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue
            
            # 所有可见的关节点的坐标的均值
            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2) # 人体框的面积
            joints_center = np.array([joints_x, joints_y]) # 所有可见的关节点的坐标的中心点
            bbox_center = np.array(rec['center']) # 人体框的中心
            # np.linalg.norm(x, ord=None) - 求范数
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2) # 求二范数
            
            # ks - keypoint similarity 即 所有可见的关节点坐标的中心点 和 人体框中心点 相似度（公式类似于OKS的计算，sigma=0.2）
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            #? 这个阈值为什么这种形式？每个可见的关节点的权重是0.2/16？那为什么还要加上0.45？
            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16 # 大于一定阈值，该阈值与num_vis有关，可见关节点越多越大
            
            #* 只有 所有可见的关节点坐标的中心点 和 人体框中心点 的相似度超过一定阈值才选择该数据
            #? 是否暗含 所有可见的关节点坐标的中心点 与 人体框中心点 重合最好？
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        """生成GT的高斯热力图

        Args:
            joints ([num_joints, 3]): 经过仿射变换以后的关节点坐标
            joints_vis ([num_joints, 3]): 关节点的可见程度

        Returns:
            _type_: target - 热力图, target_weight - 关节点可视程度(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        # GT为高斯热力图
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1], # 高
                               self.heatmap_size[0]), # 宽
                              dtype=np.float32)

            tmp_size = self.sigma * 3 # 高斯半径的大小

            for joint_id in range(self.num_joints): # 对每个关节点生成GT热力图
                feat_stride = self.image_size / self.heatmap_size # 计算heatmap上一个像素点对应原图像素点的跨度
                #! Q：为什么 +0.5？A：向上取整
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5) # 从原图坐标映射到热力图上
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                # 生成以2*tmp_size为边长的正方形，中心点为关节点坐标
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # 左上角坐标
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # 右下角坐标
                # 要求高斯分布是否有在热力图内部的，如果全都不在则认定该关节点不可见
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                # np.newaxis - 新增一个轴
                y = x[:, np.newaxis] #! Q：为什么y要新增一个轴；A：保证能够触发广播，得到一个高斯核矩阵
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                # 生成以(size//2, size//2)为中心，以tmp_size为半径的高斯核，高斯区间为[0:2*tmp_size+1,0:2*tmp_size+1]
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                # 取与Image range对应位置的高斯核
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range （在热力图大小内的正方形子集区域）
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5: #? 大于0.5？为什么不是大于0
                    # 将高斯核上的值映射到关节点坐标附近的正方形上
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
