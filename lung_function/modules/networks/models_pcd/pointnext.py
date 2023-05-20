import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from openpoints.models import build_model_from_cfg

if cfg.model.get('in_channels', None) is None:
    cfg.model.in_channels = cfg.model.encoder_args.in_channels



model = build_model_from_cfg(cfg.model).to(cfg.rank)

class get_model(nn.Module):
    def __init__(self, num_class, npoint_base=512, radius_base=40, nsample_base=64):
        super(get_model, self).__init__()
        in_channel = 4

        self.sa1 = PointNetSetAbstraction(npoint=npoint_base, radius=radius_base, nsample=nsample_base, in_channel= in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=npoint_base // 2, radius=radius_base * 2, nsample=nsample_base * 2, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=npoint_base // 4, radius=radius_base * 4, nsample=nsample_base * 4, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyzr):
        B, _, _ = xyzr.shape
      
        l1_xyz, l1_points = self.sa1(xyzr[:,:3,:], xyzr[:,3:4,:])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        x = l4_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x

