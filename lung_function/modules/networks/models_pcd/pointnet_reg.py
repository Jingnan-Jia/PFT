import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from lung_function.modules.networks.models_pcd.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, k=4, pointnet_fc_ls=None, loss=None, dp_fc1_flag=False):
        super(get_model, self).__init__()
        channel = 4
        fc_ls = pointnet_fc_ls  # with length of 2 or 3
        self.feat = PointNetEncoder(
            global_feat=True, feature_transform=True, channel=channel, feature_nb=fc_ls[0])
        self.fc1 = nn.Linear(fc_ls[0], fc_ls[1])
        self.dp_fc1_flag = dp_fc1_flag
        self.dp_fc1 = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(fc_ls[1])

        if len(fc_ls) == 3:            
            self.fc2 = nn.Linear(fc_ls[1], fc_ls[2])
            self.bn2 = nn.BatchNorm1d(fc_ls[2])
            self.dropout = nn.Dropout(p=0.4)

            self.fc3 = nn.Linear(fc_ls[2], k)
        elif len(fc_ls) == 2:
            self.fc2 = nn.Linear(fc_ls[1], k)
        else:      
            raise Exception(f"wrong fc_ls: {fc_ls}, the length should be 2 or 3")
        
        self.fc_ls = fc_ls
        # self.relu = nn.ReLU()
        self.loss = loss

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        if self.dp_fc1_flag:
            x = F.relu(self.bn1(self.dp_fc1(self.fc1(x))))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
        if len(self.fc_ls) == 3:
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            x = self.fc3(x)
        elif len(self.fc_ls) == 2:
            x = self.fc2(x)
        else:
            raise Exception(f"wrong fc_ls: {self.fc_ls}, the length should be 2 or 3")

        if self.loss == 'mse_regular':
            return x, trans_feat
        else:
            return x


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
