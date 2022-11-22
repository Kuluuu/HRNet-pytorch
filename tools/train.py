# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args) # 生成一个config文件

    """
    logger - 用于打印输出
    final_output_dir - 存放输出的文件夹
    tb_log_dir - 存放训练log文件的文件夹
    """
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args)) # 打印args参数
    logger.info(cfg) # 打印cfg文件内容

    #* cudnn related setting
    # 如果输入大小是固定的，采用cudnn.benchmark是可以加速推理和训练的。
    # 除了启动时会花额外的时间配置。但对于输入大小会变动的，每变一次就会配置一次，反而会影响速度。
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # 若设置为True，每次返回的卷积算法将是确定的，即默认算法。
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    # cuDNN使用的非肯定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__) # 返回.py文件的路径
    #* 保存网络架构到 cfg.OUTPUT_DIR / cfg.DATASET.DATASET / cfg.MODEL.NAME / cfg_name 文件下
    shutil.copy2( # 将源文件的内容复制到目标文件或目录
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), # 保存模型的.py文件
        final_output_dir)
    # logger.info(pprint.pformat(model))

    # 初始化tensorboard
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # dump_input用于测试网络模型
    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    # 输出模型的details
    logger.info(get_model_summary(model, dump_input))

    # 多卡GPU并行训练
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize( # 减去数据对应维度的统计平均值，可以消除公共部分，以凸显个体之前的差异和特征。
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] # ImageNet的均值和标准差
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    """ pin_memory
        通常情况下，数据在内存中要么以锁页的方式存在，要么保存在虚拟内存(磁盘)中。
        pin_memory设置为True后，数据直接保存在锁页内存中，后续直接传入cuda；
        否则需要先从虚拟内存中传入锁页内存中，再传入cuda，
        这样就比较耗时了，但是对于内存的大小要求比较高。
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
