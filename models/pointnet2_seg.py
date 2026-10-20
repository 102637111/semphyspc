"""PointNet2Seg: 标准 PointNet++（SSG，set abstraction + feature propagation）逐点语义分割基线。

作为 C⑤「公平基线」的两个代表性 point-based 分割模型之一（另一个是 PointTransformer），
与 PointMamba 在**同一数据协议**（16k 点 / 6 通道 XYZ+RGB / 12 类）下对比，隔离「backbone
分组方式」这一个变量：PointNet++ 用 FPS + ball query 的层次集合抽象，PointMamba 用
FPS+KNN 或 0.5m 体素网格分组。

Pipeline（经典 SSG 语义分割，特征统一用 (B, C, N) 布局，坐标用 (B, N, 3)）：
    pts (B, N, 6: XYZ+RGB)
        -> per-sample 归一化 XYZ 到零均值单位球（标准 PointNet++ 预处理）
        -> SA1(2048, r0.1) -> SA2(512, r0.2) -> SA3(128, r0.4) -> SA4(32, r0.8)
        -> FP4 -> FP3 -> FP2 -> FP1（FP1 的 skip 是原始 6 通道点特征）
        -> seg_head (B, N, num_classes)

接口与 PointMambaSeg 对齐：forward 返回 (logits, None)，提供 get_loss() 供 runner_seg 调用。

注意：PointNet++ 的 ball query 是固定半径（scale-sensitive），故在 forward 内做标准
per-sample 零均值单位球归一化；PointMamba 用 0.5m 网格在原始米制坐标上分组，两者都是各自
方法的规范操作（论文 §4.x 已注明）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .build import MODELS


class PointNetSetAbstraction(nn.Module):
    """set abstraction：FPS + ball query 分组 + 共享 MLP + max-pool（标准 (B,C,N) 布局）。"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last = out_ch

    def forward(self, xyz, points):
        """xyz (B,N,3)，points (B,C_in,N) 特征。

        返回 new_xyz (B,npoint,3)，new_points (B,C_out,npoint)。
        """
        B, N, _ = xyz.shape

        fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.npoint)  # (B,npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()   # (B,npoint,3)

        idx = pointnet2_utils.ball_query(
            self.radius, self.nsample, xyz.contiguous(), new_xyz.contiguous())          # (B,npoint,nsample)

        # 分组后的相对坐标：(B,3,npoint,nsample) -> (B,npoint,nsample,3) 减去中心
        grouped_xyz = pointnet2_utils.grouping_operation(
            xyz.transpose(1, 2).contiguous(), idx)                                      # (B,3,npoint,nsample)
        grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()                      # (B,npoint,nsample,3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)                                # 相对中心坐标

        if points is not None:
            # 分组特征：(B,C_in,npoint,nsample)，与相对坐标沿通道拼接
            grouped_points = pointnet2_utils.grouping_operation(points.contiguous(), idx)  # (B,C_in,npoint,nsample)
            new_points = torch.cat([grouped_points, grouped_xyz.permute(0, 3, 1, 2)], dim=1)  # (B,C_in+3,npoint,nsample)
        else:
            new_points = grouped_xyz.permute(0, 3, 1, 2)                                # (B,3,npoint,nsample)

        new_points = new_points.permute(0, 1, 3, 2).contiguous()                        # (B,C_in,nsample,npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0]                                    # (B,C_out,npoint)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """feature propagation：3NN 距离反比插值 + skip 拼接 + MLP。"""

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last = out_ch

    def forward(self, xyz1, xyz2, points1, points2):
        """xyz1 (B,N,3) 目标(稠)，xyz2 (B,M,3) 源(稀)；
        points1 (B,C1,N) skip 特征，points2 (B,C2,M) 待传播特征。

        返回 (B, C_out, N)。
        """
        dists, idx = pointnet2_utils.three_nn(xyz1.contiguous(), xyz2.contiguous())  # (B,N,3)
        dist_recip = 1.0 / (dists + 1e-8)
        weight = dist_recip / torch.sum(dist_recip, dim=-1, keepdim=True)
        interpolated = pointnet2_utils.three_interpolate(
            points2.contiguous(), idx, weight.contiguous())  # (B, C2, N)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=1)  # (B, C1+C2, N)
        else:
            new_points = interpolated
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


@MODELS.register_module()
class PointNet2Seg(nn.Module):
    """PointNet++（SSG）逐点语义分割，接口与 PointMambaSeg 对齐。"""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.in_channels = getattr(config, 'in_channels', 6)
        self.num_classes = config.num_classes

        np1 = getattr(config, 'sa1_npoint', 2048)
        np2 = getattr(config, 'sa2_npoint', 512)
        np3 = getattr(config, 'sa3_npoint', 128)
        np4 = getattr(config, 'sa4_npoint', 32)

        self.sa1 = PointNetSetAbstraction(np1, getattr(config, 'sa1_radius', 0.1), 32, self.in_channels + 3, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(np2, getattr(config, 'sa2_radius', 0.2), 32, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(np3, getattr(config, 'sa3_radius', 0.4), 32, 128 + 3, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(np4, getattr(config, 'sa4_radius', 0.8), 32, 256 + 3, [256, 256, 512])

        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp2 = PointNetFeaturePropagation(128 + 64, [128, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + self.in_channels, [128, 128, 128])

        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(128, self.num_classes, 1),
        )

    def _normalize_xyz(self, xyz):
        """per-sample 零均值单位球归一化（标准 PointNet++ 预处理）。"""
        centroid = xyz.mean(dim=1, keepdim=True)  # (B,1,3)
        xyz_c = xyz - centroid
        scale = xyz_c.norm(dim=-1).max(dim=1, keepdim=True).values.clamp(min=1e-6)  # (B,1,1)
        return xyz_c / scale.unsqueeze(-1)

    def forward(self, pts):
        xyz = self._normalize_xyz(pts[:, :, :3])       # (B,N,3)
        feat = pts.transpose(1, 2)                     # (B, 6, N)  XYZ+RGB 全部通道

        l1_xyz, l1_points = self.sa1(xyz, feat)        # (B,2048,3),(B,64,2048)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B,256,128)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B,128,512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B,128,2048)
        l0_points = self.fp1(xyz, l1_xyz, feat, l1_points)          # (B,128,N)  skip=原始6通道

        logits = self.seg_head(l0_points).transpose(1, 2)  # (B, N, num_classes)
        return logits, None

    def get_loss(self, ret, gt):
        B, N, C = ret.shape
        ret = ret.reshape(-1, C)
        gt = gt.reshape(-1).long()
        valid = gt >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=ret.device, requires_grad=True)
        return F.cross_entropy(ret[valid], gt[valid])

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        base = ckpt.get('base_model') or ckpt.get('model')
        base = {k.replace('module.', ''): v for k, v in base.items()}
        self.load_state_dict(base, strict=False)
        print(f'[PointNet2Seg] loaded pretrained weights from {ckpt_path}')
