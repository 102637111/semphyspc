"""PointTransformerSeg: 标准 PointTransformer（Zhao et al. ICCV 2021）逐点语义分割基线。

作为 C⑤「公平基线」的两个代表性 point-based 分割模型之一（另一个是 PointNet++），
与 PointMamba 在**同一数据协议**（16k 点 / 6 通道 XYZ+RGB / 12 类）下对比。PointTransformer
用「FPS + kNN 的 transition down + 向量自注意力 block + 插值 transition up」，是 point-based
Transformer 的代表；PointMamba 用 Mamba 序列建模。隔离的是「backbone 网络结构」这一个变量。

Pipeline（经典 PointTransformer 语义分割，5 层）：
    pts (B, N, 6) -> per-sample 归一化 XYZ（零均值单位球，标准预处理）
        -> input embed Linear(6 -> 32)
        -> TD(16384->4096, 32->64) -> Block[2]
        -> TD(4096->1024, 64->128)  -> Block[2]
        -> TD(1024->256, 128->256)  -> Block[2]
        -> TD(256->64, 256->512)    -> Block[2]
        -> TU(64->256) -> TU(256->1024) -> TU(1024->4096) -> TU(4096->16384)
        -> head Linear(32 -> num_classes)

向量自注意力：q/k/v 各自加相对位置编码 MLP(Δp)，注意力为标量 q·k/√C softmax 后对 v 加权求和。
接口与 PointMambaSeg 对齐：forward 返回 (logits, None)，提供 get_loss() 供 runner_seg 调用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .build import MODELS


def fps_idx(xyz, npoint):
    """(B, N, 3) -> (B, npoint) FPS 下标。"""
    return pointnet2_utils.furthest_point_sample(xyz.contiguous(), npoint)


def gather_xyz(xyz, idx):
    """(B, N, 3) + (B, S) -> (B, S, 3)。"""
    return pointnet2_utils.gather_operation(
        xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()


def knn_idx(xyz, query, k):
    """欧氏距离 kNN：(B, N, 3) source + (B, M, 3) query -> (B, M, k) 下标。

    与 utils/knn_pytorch.KNN 同公式（|a-b|^2 = |a|^2 + |b|^2 - 2a·b），但写成函数式便于
    每个 block / transition down 用不同 k。
    """
    xyz_sq = (xyz ** 2).sum(dim=-1)            # (B, N)
    q_sq = (query ** 2).sum(dim=-1)            # (B, M)
    cross = torch.bmm(query, xyz.transpose(1, 2))  # (B, M, N)
    dist = q_sq.unsqueeze(-1) + xyz_sq.unsqueeze(1) - 2 * cross
    dist = dist.clamp(min=0.0)
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx.int()  # pointnet2_ops 的 grouping_operation 要求 int32 下标


def index_points(points, idx):
    """按最后一维下标 gather：(B, N, C) + (B, S[, k]) -> (B, S[, k], C)。"""
    B = points.shape[0]
    view_shape = [1] * (idx.dim() - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(B, *view_shape).repeat(*repeat_shape)
    return points[batch_idx, idx, :]


class PointTransformerBlock(nn.Module):
    """向量自注意力 block（PointTransformer 核心层）。

    q 由中心点特征 + 相对位置编码得到，k/v 由邻居特征 + 相对位置编码得到；
    注意力为标量 q·k / √C，softmax 后对 v 加权求和，再残差 + FFN。
    """

    def __init__(self, dim, k=16):
        super().__init__()
        self.dim = dim
        self.k = k
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.linear_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, pos):
        B, N, C = x.shape
        idx = knn_idx(pos, pos, self.k)               # (B, N, k)，含自身（最近）
        neighbor_x = index_points(x, idx)             # (B, N, k, C)
        neighbor_pos = index_points(pos, idx)         # (B, N, k, 3)
        rel = neighbor_pos - pos.unsqueeze(2)         # (B, N, k, 3)
        pos_emb = self.pos_mlp(rel)                   # (B, N, k, C)

        q = self.linear_q(x).unsqueeze(2) + pos_emb   # (B, N, 1, C) -> (B, N, k, C)
        k = self.linear_k(neighbor_x) + pos_emb       # (B, N, k, C)
        v = self.linear_v(neighbor_x) + pos_emb       # (B, N, k, C)

        attn = (q * k).sum(dim=-1) / (C ** 0.5)       # (B, N, k) 标量注意力
        attn = F.softmax(attn, dim=-1)
        out = (v * attn.unsqueeze(-1)).sum(dim=2)     # (B, N, C)
        out = self.linear_out(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class TransitionDown(nn.Module):
    """FPS 下采样 + kNN 分组 + max-pool + MLP 升维。"""

    def __init__(self, in_dim, out_dim, npoint, k=16):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, pos):
        """x (B,N,C), pos (B,N,3) -> (B,S,out_dim), (B,S,3)。"""
        fps = fps_idx(pos, self.npoint)               # (B, S)
        new_pos = gather_xyz(pos, fps)                # (B, S, 3)
        knn = knn_idx(pos, new_pos, self.k)           # (B, S, k)
        grouped = pointnet2_utils.grouping_operation(
            x.transpose(1, 2).contiguous(), knn)      # (B, C, S, k)
        pooled = grouped.max(dim=3)[0]                # (B, C, S)
        out = self.mlp(pooled.transpose(1, 2))        # (B, S, out_dim)
        return out, new_pos


class TransitionUp(nn.Module):
    """3NN 插值上采样 + skip 拼接 + MLP。"""

    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + skip_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, pos, skip_x, skip_pos):
        """x (B,M,C_in) 待上采样特征 + pos (B,M,3)；skip_x (B,N,C_skip) + skip_pos (B,N,3)。

        返回 (B, N, out_dim)。
        """
        dists, idx = pointnet2_utils.three_nn(skip_pos.contiguous(), pos.contiguous())  # (B,N,3)
        dist_recip = 1.0 / (dists + 1e-8)
        weight = dist_recip / torch.sum(dist_recip, dim=-1, keepdim=True)
        interp = pointnet2_utils.three_interpolate(
            x.transpose(1, 2).contiguous(), idx, weight.contiguous())  # (B, C_in, N)
        out = torch.cat([interp, skip_x.transpose(1, 2)], dim=1)       # (B, C_in+C_skip, N)
        out = self.mlp(out.transpose(1, 2))                            # (B, N, out_dim)
        return out


@MODELS.register_module()
class PointTransformerSeg(nn.Module):
    """PointTransformer 逐点语义分割，接口与 PointMambaSeg 对齐。"""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.in_channels = getattr(config, 'in_channels', 6)
        self.num_classes = config.num_classes

        enc_dims = getattr(config, 'pt_dims', [32, 64, 128, 256, 512])
        npoints = getattr(config, 'pt_npoints', [4096, 1024, 256, 64])
        nblocks = getattr(config, 'pt_blocks', [2, 2, 2, 2])
        k = getattr(config, 'pt_k', 16)

        self.input_embed = nn.Linear(self.in_channels, enc_dims[0])

        # encoder：交替 TD + Block，记录各层 (pos, feat) 供 skip
        self.tds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i, npt in enumerate(npoints):
            self.tds.append(TransitionDown(enc_dims[i], enc_dims[i + 1], npt, k=k))
            self.blocks.append(nn.ModuleList(
                [PointTransformerBlock(enc_dims[i + 1], k=k) for _ in range(nblocks[i])]))

        # decoder：逐层插值回原分辨率
        self.tus = nn.ModuleList()
        # enc 层数 = len(dims) = 5：level0..4；decoder 从 level4 回到 level0
        for i in reversed(range(1, len(enc_dims))):
            # 从 enc_dims[i]（更粗）上采样到 enc_dims[i-1]（更细），skip 为 enc_dims[i-1]
            self.tus.append(TransitionUp(enc_dims[i], enc_dims[i - 1], enc_dims[i - 1]))

        self.seg_head = nn.Sequential(
            nn.Linear(enc_dims[0], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes),
        )

    def _normalize_xyz(self, xyz):
        centroid = xyz.mean(dim=1, keepdim=True)
        xyz_c = xyz - centroid
        scale = xyz_c.norm(dim=-1).max(dim=1, keepdim=True).values.clamp(min=1e-6)
        return xyz_c / scale.unsqueeze(-1)

    def forward(self, pts):
        pos = self._normalize_xyz(pts[:, :, :3])          # (B, N, 3)
        rgb = pts[:, :, 3:]
        if self.in_channels == 6 and rgb.shape[-1] > 0:
            feat_in = torch.cat([pos, rgb], dim=-1)
        else:
            feat_in = pos

        x = self.input_embed(feat_in)                     # (B, N, C0)

        # encoder，逐层记录 (pos, feat) 作为 skip
        skips_pos = [pos]
        skips_feat = [x]
        cur_pos, cur_x = pos, x
        for td, blks in zip(self.tds, self.blocks):
            cur_x, cur_pos = td(cur_x, cur_pos)
            for blk in blks:
                cur_x = blk(cur_x, cur_pos)
            skips_pos.append(cur_pos)
            skips_feat.append(cur_x)

        # decoder：从最粗层往上插值
        x = skips_feat[-1]                                # level4 特征
        pos_ = skips_pos[-1]                              # level4 位置
        for i, tu in enumerate(self.tus):
            skip_level = len(skips_feat) - 2 - i          # 从 level3 开始往下
            x = tu(x, pos_, skips_feat[skip_level], skips_pos[skip_level])
            pos_ = skips_pos[skip_level]

        logits = self.seg_head(x)                         # (B, N, num_classes)
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
        print(f'[PointTransformerSeg] loaded pretrained weights from {ckpt_path}')
