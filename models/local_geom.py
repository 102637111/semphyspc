"""局部几何旁路分支：ball-query 半径邻域 + 相对坐标 MLP，输出逐点几何特征。

目标：给罕见类/小物体补充"局部几何结构"信息（相对坐标 + 邻域距离），
弥补 GridGroup 分组时小物体点被吞并、以及 group 级 token 丢失逐点邻域的问题。

第一版用 edge-conv 式（最成熟可靠）：半径邻域 -> 相对坐标 -> 小 MLP -> max-pool。
用 pointnet2_ops.ball_query（半径邻域）而非 utils/knn_pytorch.py 的 O(N²) KNN：
后者会物化完整 pairwise 距离矩阵，16k 点上直接 OOM。
"""
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils


class LocalGeomBranch(nn.Module):
    def __init__(self, radius=0.5, nsample=32, out_dim=128):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        # 输入：相对坐标(3) + 邻域距离(1) = 4 维
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 1),
        )

    def forward(self, xyz):
        # xyz: (B, N, 3)
        # 自查询：每个点作为自己的 query center，取半径内 nsample 个邻居。
        # ball_query 返回 int32 下标 (B,N,nsample)，grouping_operation 直接用。
        idx = pointnet2_utils.ball_query(
            self.radius, self.nsample, xyz.contiguous(), xyz.contiguous())  # (B,N,nsample)
        xyz_t = xyz.transpose(1, 2).contiguous()                            # (B,3,N)
        grouped = pointnet2_utils.grouping_operation(xyz_t, idx)            # (B,3,N,nsample)
        center = xyz_t.unsqueeze(-1)                                        # (B,3,N,1)
        rel = grouped - center                                              # 相对坐标 (B,3,N,nsample)
        dist = rel.norm(dim=1, keepdim=True)                                # (B,1,N,nsample)
        feat = torch.cat([rel, dist], dim=1)                                # (B,4,N,nsample)
        feat = self.mlp(feat)                                               # (B,out_dim,N,nsample)
        feat = feat.max(dim=-1)[0]                                          # (B,out_dim,N)
        return feat
