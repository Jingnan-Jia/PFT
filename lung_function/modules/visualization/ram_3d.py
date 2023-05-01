import os
import sys
from torchsummary import summary

# sys.path.append("../..")
sys.path.append("../../..")
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from tqdm import tqdm
from mlflow import log_metric, log_metrics, log_param, log_params
import mlflow
from mlflow.tracking import MlflowClient
from medcam3d import GradCAM
from medutils.medutils import load_itk, save_itk

from lung_function.modules.path import PFTPath
from lung_function.modules.datasets import all_loaders
# from lung_function.modules.cam import GradCAM
from lung_function.scripts.run import Run
from lung_function.modules.set_args import get_args

class Args:  # convert dict to class attribute
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                else:
                    try:
                        value = float(value)  # convert to float value if possible
                        try:
                            if int(value) == value:  # convert to int if possible
                                value = int(value)
                        except Exception:
                            pass
                    except Exception:
                        pass
                setattr(self, key, value)
                
class RegressionOutputTarget:  # devine my custom target
    def __init__(self, target_output, all_outputs):

        if target_output in all_outputs:
            self.target_out_idx = all_outputs.index(target_output)
        else:
            raise Exception(f"all_outputs is {all_outputs}, but you wanted {target_output} from them")


    def __call__(self, model_output):
        if len(model_output.shape) == 1:  # shape: (N,), A vector
            if model_output.shape[0] == 1:
                return model_output  # TODO: for the single output network
            else:
                return model_output[self.target_out_idx]  # return one node
        
        # shape: (M, N), A 2d array
        return model_output[:, self.category]  # return a vector 

def get_loader(mode, mypath, max_img_nb, args):
    label_all = pd.read_csv(mypath.save_label_fpath(mode))
    pred_all = pd.read_csv(mypath.save_pred_fpath(mode))
    mae_all = (label_all - pred_all).abs()
    mae_all['average'] = mae_all.mean(numeric_only=True, axis=1)
    label_all_sorted = label_all.loc[mae_all['average'].argsort()[:max_img_nb]]
    top_pats = label_all_sorted['pat_id'].to_list()

    data_dt = all_loaders(mypath.data_dir, mypath.label_fpath, args, datasetmode='valid', top_pats=top_pats)
    dataloader = data_dt['valid']
    return dataloader

def main():
    # update parameters
    AttentionMethod = "GradCAM"  # or others
    Ex_id = 2666  # 2522 vgg 4-out, 2601 for vgg, 2657 for x3d_m FEV, 2666 for x3d_m of four outputs.
    max_img_nb = 3
    mode = 'valid'
    
    # retrive the run for the ex id
    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment("lung_fun_db15")
    client = MlflowClient()
    run_ls = client.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=f"params.id LIKE '%{Ex_id}%'")
    run_ = run_ls[0]
    args_dt = run_.data.params  # extract the hyper parameters

    args = Args(args_dt)  # convert to object
    args.workers=1

    args.use_cuda = True
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    mypath = PFTPath(Ex_id, check_id_dir=False, space=args.ct_sp)

    args.pretrained_id = Ex_id
    myrun = Run(args, dataloader_flag=False)

    if args.net=='x3d_m':
        target_layers = [
            # myrun.net.blocks[1].res_blocks[0].branch2.conv_b,
                        #  myrun.net.blocks[2].res_blocks[0].branch2.conv_b,
        #                  myrun.net.blocks[3].res_blocks[0].branch2.conv_b,
                        #  myrun.net.blocks[4].res_blocks[0].branch2.conv_b,
                         myrun.net.blocks[5].proj,
                         ]  # TODO: change this line select which layer
    else:
        target_layers = [
                        
                    
                        # all pooling layers
                        # myrun.net.avgpool,
                        # myrun.net.features[28], 
                        myrun.net.features[21], 
                        # myrun.net.features[14], 
                        # myrun.net.features[7], 
                        # myrun.net.features[3], 

                        #  myrun.net.features[18], 
                        #  myrun.net.features[15], 
                        ]  # TODO: change this line select which layer


    # select the top accurate patients
    
    dataloader = get_loader(mode=mode, mypath=mypath, max_img_nb=max_img_nb, args=args)


    if '-' in args.target:
        all_outputs_ls = args.target.split('-')  # possible names: DLCOc, FEV1, FVC, TLC, DLCOcPP, FEV1PP, FVCPP, TLCPP
    else:
        all_outputs_ls = [args.target]


    for data in dataloader:

        batch_pat_id = data['pat_id'].detach().numpy()
        batch_x = data[args.input_mode][:,:,:,:,:]  # ct  ct_masked_by_torso
        batch_y = data['label']
        batch_ori = data['origin'].detach().numpy()
        batch_sp = data['spacing'].detach().numpy()
        
        
        for pat_id, image, ori, sp, label in zip(batch_pat_id, batch_x, batch_ori, batch_sp, batch_y):

            ct_fpath = f"{myrun.mypath.id_dir}/cam/SSc_patient_{pat_id[0]}.mha"
            save_itk(ct_fpath, image[0].detach().numpy(), np.float64(ori), np.float64(sp), dtype='float')

            img = image[None].to(device)
            
                
            for target_output in all_outputs_ls:
                targets = [RegressionOutputTarget(target_output = target_output, all_outputs = all_outputs_ls)]  # TODO: change it to select which output
                cam = GradCAM(model=myrun.net, target_layers=target_layers, use_cuda=args.use_cuda)

                    
                grayscale_cam = cam(input_tensor=img, targets=targets)  # shape: 1, z, y, x
                cam_fpath = f"{myrun.mypath.id_dir}/cam/SSc_patient_{pat_id[0]}_target_{target_output}_pool4_blocks[4].res_blocks[0].branch2.conv_b.mha"
                save_itk(cam_fpath, grayscale_cam[0], np.float64(ori), np.float64(sp), dtype='float')
                # attention.run(pat_id, image, ori, sp, label)  # for my own code of RAM
                print('Finish pat_id: ', pat_id[0],target_output )
    print("Finish all")

if __name__ == '__main__':
    main()