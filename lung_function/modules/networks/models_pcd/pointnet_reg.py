import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=4, pointnet_fc_ls=None, loss=None):
        super(get_model, self).__init__()
        channel = 4
        fc_ls = pointnet_fc_ls
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel, feature_nb=fc_ls[0])
        self.fc1 = nn.Linear(fc_ls[0], fc_ls[1])
        self.fc2 = nn.Linear(fc_ls[1], fc_ls[2])
        self.fc3 = nn.Linear(fc_ls[2], k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(fc_ls[1])
        self.bn2 = nn.BatchNorm1d(fc_ls[2])
        self.relu = nn.ReLU()
        self.loss = loss

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        if self.loss=='mse_regular':
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
