# -*- coding: utf-8 -*-
# @Time    : 7/5/21 9:27 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from .cnn_fc3d import Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d, Vgg16_3d, Vgg19_3d
from .cnn_fc3d_enc import Cnn3fc1Enc, Cnn3fc2Enc, Cnn4fc2Enc, Cnn5fc2Enc, Cnn6fc2Enc, Vgg11_3dEnc
from .vit3 import ViT3
from torch import nn
import torch
import torchvision
import importlib
import sys
sys.path.append("models_pcd")
from lung_function.modules.openpoints.models import build_model_from_cfg
from lung_function.modules.openpoints.utils import EasyConfig
from mlflow import log_params

def get_net_3d(name: str,
               nb_cls: int,
               fc1_nodes=1024, 
               fc2_nodes=1024, 
               image_size=240, 
               pretrained=True, 
               pointnet_fc_ls=None,
               loss=None,
               dp_fc1_flag=False, 
               args=None):
    level_node = 0
    if 'pointnet' in name:
        # if name=='pointnet_reg':
        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find('ReLU') != -1:
                m.inplace = True
        pcd_model = importlib.import_module(name)
        if name=='pointnet_reg':  

            net = pcd_model.get_model(nb_cls, pointnet_fc_ls, loss, dp_fc1_flag)
        else:  # pointnet++
            net = pcd_model.get_model(nb_cls, npoint_base=args.npoint_base, radius_base=args.radius_base, nsample_base=args.nsample_base)
        net.apply(inplace_relu)
        # elif name=='pointnet2_reg':
    elif 'pointnext' in name:
        
        cfg = EasyConfig()
        cfg_fpath = "/home/jjia/data/lung_function/lung_function/modules/cfgs/" + args.cfg

        cfg.load(cfg_fpath, recursive=True)  # args.cfs is the path of the cfg file
        cfg.radius = args.radius_base
        cfg.radius_scaling = args.radius_scaling
        cfg.sa_layers = args.sa_layers
        cfg.nsample = args.nsample_base
        cfg.num_classes = nb_cls
        net = build_model_from_cfg(cfg.model)  # pass a config set to this function to build a model

    elif name == 'cnn3fc1':
        net = Cnn3fc1(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                      num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn3fc2':
        net = Cnn3fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                      num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn4fc2':
        net = Cnn4fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                      num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn5fc2':
        net = Cnn5fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                      num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn6fc2':
        net = Cnn6fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                      num_classes=nb_cls, level_node=level_node)
    elif name == "vgg11_3d":
        net = Vgg11_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                       num_classes=nb_cls, level_node=level_node)
    elif name == "vgg16_3d":
        net = Vgg16_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                       num_classes=nb_cls, level_node=level_node)
    elif name == "vgg19_3d":
        net = Vgg19_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                       num_classes=nb_cls, level_node=level_node)
    elif name == "vit3":
        net = ViT3(dim=1024, image_size=image_size, patch_size=20,
                   num_classes=nb_cls, depth=6, heads=8, mlp_dim=2048, channels=1)
    elif name in ("r3d_18", "r2plus1d_18"):
        # class NewStem(nn.Sequential):
        #     """The new conv-batchnorm-relu stem"""
        #     def __init__(self) -> None:
        #         super().__init__(
        #             nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
        #             nn.BatchNorm3d(64),
        #             nn.ReLU(inplace=True),
        #         )
        if name == "r3d_18":
            net = torchvision.models.video.r3d_18(
                pretrained=pretrained, progress=True)
        else:
            net = torchvision.models.video.r2plus1d_18(
                pretrained=pretrained, progress=True)
        # net.stem = NewStem()
        net.stem[0] = nn.Conv3d(1, 64, kernel_size=(
            3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        net.fc = torch.nn.Linear(512 * 1, nb_cls)
    elif name in ('slow_r50', 'slowfast_r50', "x3d_xs", "x3d_s", "x3d_m", "x3d_l"):
        if name == "slow_r50":
            # Christoph et al, “SlowFast Networks for Video Recognition” https://arxiv.org/pdf/1812.03982.pdf
            net = torch.hub.load(
                'facebookresearch/pytorchvideo', name, pretrained=pretrained)
            net.blocks[0].conv = nn.Conv3d(1, 64, kernel_size=(
                1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        elif name == "slowfast_r50":
            net = torch.hub.load(
                'facebookresearch/pytorchvideo', name, pretrained=pretrained)
            net.blocks[0].multipathway_blocks[0].conv = nn.Conv3d(
                1, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
            net.blocks[0].multipathway_blocks[1].conv = nn.Conv3d(
                1, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        else:
            net = torch.hub.load(
                'facebookresearch/pytorchvideo', name, pretrained=pretrained)
            net.blocks[0].conv.conv_t = nn.Conv3d(1, 24, kernel_size=(
                1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        net.blocks[-1].proj = nn.Linear(in_features=2048,
                                        out_features=nb_cls, bias=True)
        # net.blocks[-1].output_pool = nn.Linear(in_features=400, out_features=nb_cls, bias=True)
    else:
        raise Exception('wrong net name', name)

    return net


def get_net_pos_enc(name: str, nb_cls: int, fc1_nodes=1024, fc2_nodes=1024, level_node=0):
    if name == 'cnn3fc1':
        net = Cnn3fc1Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                         num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn3fc2':
        net = Cnn3fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                         num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn4fc2':
        net = Cnn4fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                         num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn5fc2':
        net = Cnn5fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                         num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn6fc2':
        net = Cnn6fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                         num_classes=nb_cls, level_node=level_node)
    elif name == "vgg11_3d":
        net = Vgg11_3dEnc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes,
                          num_classes=nb_cls, level_node=level_node)
    else:
        raise Exception('wrong net name', name)

    return net
