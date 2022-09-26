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


mypath = PFTPath(eval_id, check_id_dir=False, space=args_dt['ct_sp'])
device = torch.device("cuda")
target = [i.lstrip() for i in args_dt['target'].split('-')]
net = get_net_3d(name=args_dt['net'], nb_cls=len(target), image_size=args_dt['x_size']) # output FVC and FEV1

grad_block = []
fmap_block = []

net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
net.to(device)
net.ln3.register_forward_hook(farward_hook)
net.ln3.register_backward_hook(backward_hook)
net.eval()
opt = torch.optim.Adam(net.parameters(), lr=0.0001)

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

def run(self, pat_id, image: torch.Tensor, ori: np.ndarray, sp: np.ndarray, label: torch.Tensor):

    chn, w, h, d = image.shape
    img = image[None]

    # img_id = int(self.img_fpath.split('/')[-2].split('_')[-1])
    # label_all = label
    # loss = nn.MSELoss()

    output = self.net(img)
    pred_dt = {k: v for k, v in zip(self.target, output)}
    print(f"predict: {output.detach().cpu().numpy()}, label: {label.detach().cpu().numpy()}")

    for target in self.target:
        print(f"For target: {target}")
        self.grad_block = []
        self.opt.zero_grad()  # clear the gradients
        pred_dt[target].backward(retain_graph=True)
        grad = torch.stack(self.grad_block, 0)
        grads_mean = torch.mean(grad, [1, 2])

        cam = torch.zeros(list(self.fmap_block[0].shape))
        for weight, map in zip(grads_mean, self.fmap_block):
            cam += weight * map
        cam = cam[0]
        cam = torch.mean(cam, 0)
        cam = cam.cpu().detach().numpy()

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cam.reshape((w, h, d))

        if not os.path.isdir(self.mypath.id_dir + '/cam'):
            os.makedirs(self.mypath.id_dir + '/cam')
        save_itk(f"{self.mypath.id_dir}/cam/{pat_id}_{target}.nii.gz", cam, ori, sp)


def main():
    print("finish all")


if __name__ == "__main__":
    main()
