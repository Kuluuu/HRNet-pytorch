# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    """获得迭代优化器

    Args:
        cfg (_type_): cfg文件
        model (_type_): 模型

    Returns:
        _type_: 迭代优化器
    """
    optimizer = None
    # 由cfg文件决定采用什么优化器
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM, # 动量参数
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV # 是否采用nesterov动量SGD
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """获取模型的概要（参数量、计算量、模型网络层数），verbose时显示每层的详细信息

    Args:
        model (nn.Module): 模型
        item_length (int, optional): 调整信息间的空格数. Defaults to 26.
        verbose (bool, optional): 是否输出每层的详细信息. Defaults to False.

    Returns:
        str: 模型details
    """

    summary = [] # 存放ModuleDetails

    # 包含（层的名字，输入尺寸，输出尺寸，参数个数，计算量）
    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {} # 元组，用于记录网络层类型对应的个数

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            #* 统计各种层的个数
            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index # 没有这种层，则建立一个层
            else:
                instance_index = layer_instances[class_name] + 1 # 个数加一
                layer_instances[class_name] = instance_index

            #* 显示该层类型的第多少个
            layer_name = class_name + "_" + str(instance_index) 

            #* 统计参数量
            params = 0
            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1: # 若层为Conv或BatchNorm或Linear
                for param_ in module.parameters(): # 读取参数
                    params += param_.view(-1).size(0) # 参数个数kernel_size**2 * c_in * c_out

            flops = "Not Available" # 无需计算的层
            
            #* 统计计算量
            # 计算Conv层的计算量（Conv层，且该层中有weight属性）
            if class_name.find("Conv") != -1 and hasattr(module, "weight"): # hasattr() 函数用于判断对象是否包含对应的属性。
                flops = (
                    torch.prod( # 返回 input 张量中所有元素的乘积。
                        # module.weight.data.size() - c_out * c_in * k * k
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        # list(output.size())[2:] - 该层输出的特征图尺寸
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear): # 计算线性层的计算量
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops
                    )
            )
        # 非ModuleList，且非Sequential，且module != model
        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook)) # 主要的作用是在不改变torch网络的情况下获取某一层的输出

    model.eval()
    model.apply(add_hooks) # 统计模型信息

    space_len = item_length # 空格的个数（为了显示美观）

    #? 为什么要空跑模型，并且移除hook？
    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    #* 统计总的参数量和计算量
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    
    # os.linesep换行符
    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details
