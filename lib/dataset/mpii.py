# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    """MPII数据集
    其中：0 - r ankle, 1 - r knee, 2 - r hip,3 - l hip,4 - l knee, 5 - l ankle, 6 - l ankle， 7 - l ankle，8 - upper neck, 9 - head top,10 - r wrist,11 - r elbow, 12 - r shoulder, 13 - l shoulder,14 - l elbow, 15 - l wrist

    Args:
        JointsDataset (_type_): _description_
    """
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16
        # 配对的关节，(0-5 rl ankle, 1-4 rl knee, 2-3 rl hip, 10-15 rl wrist, 11-14 rl elbow, 12-13 rl shoulder)
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA: #? 有什么用
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db))) # 显示加载的样本数

    def _get_db(self):
        """
        对坐标进行原点为(0,0)的转换，获取数据集的GT

        Returns:
            list: [{'image', 'center', 'scale', 'joints_3d', 'joints_3d_vis', 'filename', 'imgnum'},...]
        """
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = [] # 存放处理好的GT
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            # mpii的scale属性为图片缩放率，scale=图片中人体框的高度/200，人体框为正方形的，就是原图像框图bbox范围（就是包括的人，相当于裁剪）。
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1: # 中心点坐标合法
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1, 坐标从1开始，因此转为从0开始的坐标系
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test': # test无关节点位置的GT
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1 # 将所有关节点坐标原点转换为(0, 0)
                joints_vis = np.array(a['joints_vis'])
                # 提示annot中的关节点个数与数据集不符（assert后的式子为True，则向下执行，否则打印后面的内容）
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name), # 图片路径
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
