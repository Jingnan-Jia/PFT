# -*- coding: utf-8 -*-
# @Time    : 4/15/21 10:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import numpy as np
from medutils.medutils import load_itk, save_itk
import os
from lung_function.modules.path import PFTPath
from lung_function.modules.networks import get_net_3d
import cv2
from monai.transforms import Resize


class GradCAM():
    def __init__(self, eval_id, args_dt):
        self.mypath = PFTPath(eval_id, check_id_dir=False, space=args_dt['ct_sp'])
        self.device = torch.device("cuda")
        self.target = [i.lstrip() for i in args_dt['target'].split('-')]
        self.net = get_net_3d(name=args_dt['net'], nb_cls=len(self.target), image_size=args_dt['x_size']) # output FVC and FEV1
        print('net:', self.net)

        self.grad_block = []
        self.fmap_block = []

        self.net.load_state_dict(torch.load(self.mypath.model_fpath, map_location=self.device))
        self.net.to(self.device)
        self.net.avgpool.register_forward_hook(self.farward_hook)
        self.net.avgpool.register_backward_hook(self.backward_hook)
        self.net.eval()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def run(self, pat_id, image: torch.Tensor, ori: np.ndarray, sp: np.ndarray, label: torch.Tensor):

        chn, w, h, d = image.shape
        img = image[None].to(self.device)

        # img_id = int(self.img_fpath.split('/')[-2].split('_')[-1])
        # label_all = label
        # loss = nn.MSELoss()

        output = self.net(img)
        pred_dt = {k: v for k, v in zip(self.target, output.flatten())}
        print(f"predict: {output.detach().cpu().numpy()}, label: {label.detach().cpu().numpy()}")

        for target in self.target:

            print(f"For target: {target}")
            self.grad_block = []
            self.opt.zero_grad()  # clear the gradients
            pred_dt[target].backward(retain_graph=True)
            grad = torch.stack(self.grad_block, 0)
            grads_mean = torch.mean(grad, [1, 2])

            cam = torch.zeros(list(self.fmap_block[0].shape)).to(self.device)
            for weight, map in zip(grads_mean, self.fmap_block):
                cam += weight * map
            cam = cam / len(grads_mean)
            cam = cam.cpu().detach().numpy()
            cam = np.mean(cam, 1)

            up_sp = Resize(spatial_size=(w, h, d), mode='trilinear')
            cam = up_sp(cam)[0]  # exclude the channel dim
            cam = (cam - cam.min()) / (cam.max() - cam.min()) * 256

            if target == "DLCOc/pred":
                target = "DLCOc_pred"
            if target == "FEV1/pred":
                target = "FEV1_pred"
            if target == "FVC/predNew":
                target = "FVC_predNew"
            if target == "TLC/pred":
                target = "TLC_pred"


            if not os.path.isdir(self.mypath.id_dir + '/cam'):
                os.makedirs(self.mypath.id_dir + '/cam')
            save_itk(f"{self.mypath.id_dir}/cam/{str(pat_id[0])}_{target}.mha", cam, ori.tolist(), sp.tolist())
            print(f"finish {self.mypath.id_dir}/cam/{pat_id}_{target}.nii.gz")


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def main():
    print("finish all")


if __name__ == "__main__":
    main()
