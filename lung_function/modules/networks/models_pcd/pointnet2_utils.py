import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, d]
        dst: target points, [B, M, d]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    max_pos = max(torch.max(src), torch.max(dst))
    src = src/max_pos  # I scale it to avoid 'CUDA OVERFLOW ERROR' 
    dst = dst/max_pos

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    dist = dist * max_pos

    return dist


def index_points(points, idx):
    """
    I have to add a assert check to ensure the idx does not exceed the limitation of points!
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [S1]]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]  #  batch
    view_shape = list(idx.shape)   #  [B, S, [S1]]
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B, 1]
    repeat_shape = list(idx.shape)     #  [B, S, [S1]]
    repeat_shape[0] = 1     #  [1, S, [S1]]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)   #  [B, S, [S1]]
    # points = points.to('cpu')
    # batch_indices = batch_indices.to('cpu')
    # idx = idx.to('cpu')
    # idx[idx>=points.shape[1]] = points.shape[1]-1  # test if the works
    new_points = points[batch_indices, idx, :]  # [B, S, [S1], C]
    # new_points = new_points.to(device)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, d=3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device  # could be cpu or gpu
    B, N, C = xyz.shape  # batch, number of total points, channel or feasure number (normally is 3 for xyz)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) 
    # (B, npoint) is the shape of output, respresenting the coordinates of sampled points

    distance = torch.ones(B, N).to(device) * 1e10  # will store the distance between points to ?
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  
    # shape of (B,), randomly select one point as the first point of FPS for each point set,
    # like: [88, 34, 103] when batch size is 3
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # (B,), like [0,1,2] when batch size is 3 
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



def farthest_point_sample_with_r(xyzr, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = xyzr[:,:,:3]
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query finds all points that are within a radius to the query point (an upper limit of nsample is set in implementation)
    Note: I have to add a assert check to ensure the idx does not exceed the limitation of points!
    Input:
        radius: local region radius
        nsample: **max** sample number in local region
        xyz: all points, [B, N, d=3]
        new_xyz: query points, [B, S, d=3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # shape [B, S, N] ?

    sqrdists = square_distance(new_xyz, xyz)  #  [B, S, N]
    group_idx[sqrdists > radius ** 2] = N  # distances greater than r^2 are 
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # sort_dis,group_idx=sqrdists.sort(dim=-1)
    # group_idx[sort_dis > radius ** 2] = N
    # group_idx=group_idx[:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: number of total points after `farthest_point_sample`, used as the centroid for grouping
        radius: radius value to generate new group of points
        nsample: number of point for each group after sampling
        xyz: input points position data, [B, N, d=3]
        points: input points data, [B, N, C], C is the feature number
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, d=3]
        new_points: sampled points data, [B, npoint, nsample, d+C]
    """
    B, N, d = xyz.shape
    S = npoint
    # Sampling
    fps_idx = farthest_point_sample(xyz, npoint) # index used to get value from xyz, [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # get new xyz from xyz using the index, [B, npoint, d=3]

    # Grouping
    # Ball query finds all points that are within a radius to the query point (an upper limit of nsample is set in implementation)
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # index used to get value from xyz, [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, d]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, d)  # get the relative coordinates
    grouped_xyz_norm = grouped_xyz_norm / radius # get the normalized relative coordinates, like PointNeXt

    if points is not None:
        grouped_points = index_points(points, idx) # [B, npoint, nsample, C]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, d], d=3 for 3-dim points
        points: input points data, [B, N, C], C is the feature number
    Return:
        new_xyz: sampled points position data, [B, 1, d]
        new_points: sampled points data, [B, 1, N, d+C]
    """
    device = xyz.device
    B, N, d = xyz.shape  # batch, number of points, d-min
    new_xyz = torch.zeros(B, 1, d).to(device)  # merge all points to one point, initialize the resulting coordinates to 0,0,0 for each point of each batch
    grouped_xyz = xyz.view(B, 1, N, d)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # convert poinst from [B,N,C] to [B,1,N,C+d]
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # from B,d,N to B,N,d where B is batch, d is dimension, N is the number of points per image
        if points is not None:
            points = points.permute(0, 2, 1)  # from B,C,N to B,N,C

        if self.group_all:  # group all input (grou number = 1), ususally used in the final layer of feature extraction before fully connected layer
            new_xyz, new_points = sample_and_group_all(xyz, points)
            # new_xyz: sampled points position data, [B, npoint=1, d]
            # new_points: sampled points data, [B, npoint=1, nsample=N, C+d]
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)  # sample and group, 
            # new_xyz: sampled points position data, [B, npoint, d]
            # new_points: sampled points data, [B, npoint, nsample, C+d]
        new_points = new_points.permute(0, 3, 2, 1) # to [B, C+D, nsample,npoint], if 'sample_and_group_all', npoint=1
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

