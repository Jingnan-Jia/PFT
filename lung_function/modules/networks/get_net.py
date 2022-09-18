# -*- coding: utf-8 -*-
# @Time    : 7/5/21 9:27 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from .cnn_fc3d import Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d, Vgg16_3d, Vgg19_3d
from .cnn_fc3d_enc import Cnn3fc1Enc, Cnn3fc2Enc, Cnn4fc2Enc, Cnn5fc2Enc, Cnn6fc2Enc, Vgg11_3dEnc
from .vit3 import ViT3

def get_net_3d(name: str, nb_cls: int, fc1_nodes=1024, fc2_nodes=1024,image_size=240):
    level_node = 0
    if name == 'cnn3fc1':
        net = Cnn3fc1(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn3fc2':
        net = Cnn3fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn4fc2':
        net = Cnn4fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn5fc2':
        net = Cnn5fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn6fc2':
        net = Cnn6fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vgg11_3d":
        net = Vgg11_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vgg16_3d":
        net = Vgg16_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vgg19_3d":
        net = Vgg19_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vit3":
        net =  ViT3(dim=1024, image_size=image_size, patch_size=20, num_classes=nb_cls, depth=6, heads=8, mlp_dim=2048, channels=1)
    else:
        raise Exception('wrong net name', name)

    return net


def get_net_pos_enc(name: str, nb_cls: int, fc1_nodes=1024, fc2_nodes=1024, level_node = 0):
    if name == 'cnn3fc1':
        net = Cnn3fc1Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn3fc2':
        net = Cnn3fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn4fc2':
        net = Cnn4fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn5fc2':
        net = Cnn5fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn6fc2':
        net = Cnn6fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vgg11_3d":
        net = Vgg11_3dEnc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    else:
        raise Exception('wrong net name', name)

    return net