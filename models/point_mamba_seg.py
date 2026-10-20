"""PointMambaSeg: per-point semantic segmentation on top of PointMamba.

Reuses the classification backbone (Group / dual-stream color Encoder /
MixerModel Mamba blocks / Hilbert serialization) and adds a lightweight
upsampling decoder + per-point head.

Pipeline:
    pts (B, N, 6: XYZ+RGB)
        -> Group            : neighborhood (B,G,M,6), center (B,G,3)
        -> Encoder (dual)   : group tokens (B,G,encoder_dims)
        -> pos_embed + Hilbert serialization
        -> MixerModel       : x (B,G,trans_dim)  [serialized order]
        -> un-serialize (inverse_order)
        -> KNN inverse-distance upsample G -> N
        -> concat per-point low-level feature
        -> seg_head         : (B,N,num_classes)

Supports in_channels=3 (XYZ) or 6 (XYZ+RGB).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODELS
from .point_mamba_scan import (
    Encoder,
    Group,
    MixerModel,
    serialization_func,
    apply_OrderScale,
    init_OrderScale,
)
from utils.knn_pytorch import KNN


@MODELS.register_module()
class PointMambaSeg(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        # 3 = XYZ, 6 = XYZ+RGB
        self.in_channels = getattr(config, 'in_channels', 3)
        # number of semantic classes
        self.num_classes = config.num_classes

        # ---- shared backbone (reused from classification) ----
        # 方案 B: GridSampling + voxel-patch 分组（O(N)，去掉 FPS+KNN 的 O(B*G*N) 距离矩阵）
        # use_grid_group=True 时用 GridGroup 替换 Group，并配套 O(N) 的 inverse-gather 上采样。
        self.use_grid_group = getattr(config, 'use_grid_group', False)
        self.grid_size = getattr(config, 'grid_size', 0.5)
        if self.use_grid_group:
            from .grid_group import GridGroup
            self.group_divider = GridGroup(
                num_group=self.num_group, group_size=self.group_size, grid_size=self.grid_size)
        else:
            self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.rgb_deep = getattr(config, 'rgb_deep', False)
        self.rgb_film = getattr(config, 'rgb_film', False)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, in_channels=self.in_channels,
                               rgb_deep=self.rgb_deep, rgb_film=self.rgb_film)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        drop_path = 0. if not hasattr(config, "drop_path") else config.drop_path
        rms_norm = False if not hasattr(config, "rms_norm") else config.rms_norm
        drop_out = 0. if not hasattr(config, "drop_out") else config.drop_out
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]
        mamba_version = str(getattr(config, 'mamba_version', '1'))
        mixer_type = 'mamba2' if mamba_version == '2' else 'mamba1'
        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 mixer_type=mixer_type,
                                 rms_norm=rms_norm,
                                 drop_out=drop_out,
                                 drop_path=dpr,
                                 rgb_film=self.rgb_film)

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

        # ---- segmentation decoder ----
        # low-level per-point feature branch (skip connection)
        self.point_branch = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # KNN for group-token -> point upsampling
        self.upsample_k = getattr(config, 'upsample_k', 3)
        self.knn_upsample = KNN(k=self.upsample_k, transpose_mode=True)

        # per-point semantic head: [trans_dim + 128] -> num_classes
        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim + 128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.num_classes, 1),
        )

        # per-point material head (multi-task, 与语义头共享 encoder/feat):
        # 同一 shared feature -> 独立 MLP -> 材质类别数 (UR-MAT 14 类)
        # cascade_material=True 时：材质头额外拼接"语义概率"(detach) 作为先验
        # (级联：先语义→再材质)，否则为并行头（消融基线）。
        self.num_materials = getattr(config, 'num_materials', 0)
        self.cascade_material = getattr(config, 'cascade_material', False)
        if self.num_materials > 0:
            mat_in = self.trans_dim + 128
            if self.cascade_material:
                mat_in += self.num_classes
            self.material_head = nn.Sequential(
                nn.Conv1d(mat_in, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Conv1d(256, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, self.num_materials, 1),
            )

        # instance head (optional): per-point embedding for grouping
        self.num_instances = getattr(config, 'num_instances', 0)
        if self.num_instances > 0:
            self.instance_head = nn.Sequential(
                nn.Conv1d(self.trans_dim + 128, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, self.num_instances, 1),
            )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        # P3：语义分支的类别平衡损失。config.sem_class_weight 给 32 类各一个权重，
        # 用来把监督信号从「车/可行驶面」这类海量类搬到只有几十个实例的罕见类上。
        # 只作用于类别数正好等于权重长度的那一路（即语义头），材料头仍用无权重的 CE。
        self.loss_ce_sem = None
        w = getattr(self.config, 'sem_class_weight', None)
        if w is not None:
            w = torch.tensor([float(x) for x in w], dtype=torch.float32)
            self.register_buffer('sem_class_weight', w)
            self.loss_ce_sem = nn.CrossEntropyLoss(weight=w)
        # Focal Loss：config.focal_gamma 为 None/0 时退化为标准 CE；>0 时用 Focal。
        # 动机与 P3 类平衡不同：Focal 不下调罕见类的权重，而是抑制「易分样本」的梯度，
        # 对极长尾（nuScenes IF=1670）理论上比 inverse-frequency 加权更温和。
        self.focal_gamma = getattr(self.config, 'focal_gamma', None)

    def upsample(self, group_feats, center, pts):
        """KNN inverse-distance-weighted upsample of group tokens to points.

        Args:
            group_feats: (B, G, C) token features at group centers
            center:      (B, G, 3) group center coordinates
            pts:         (B, N, C_in) original points (XYZ in first 3 channels)

        Returns:
            (B, N, C) upsampled features
        """
        B, G, C = group_feats.shape
        N = pts.shape[1]

        # KNN: for each of N points (query), find upsample_k nearest among G centers
        dists, idx = self.knn_upsample(center, pts[:, :, :3])  # (B,N,k), (B,N,k)

        # gather group features
        idx_flat = idx.view(B, -1)  # (B, N*k)
        gathered = group_feats.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, C))  # (B, N*k, C)
        gathered = gathered.view(B, N, self.upsample_k, C)

        # inverse-distance weights (dists are squared; use sqrt)
        weights = 1.0 / torch.sqrt(dists.clamp(min=1e-8))  # (B, N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize
        up = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C)
        return up

    def grid_upsample(self, group_feats, inverse):
        """O(N) upsampling: each point copies its own token's feature.

        Unlike KNN inverse-distance upsample (O(B*N*G)), this reuses the
        deterministic point -> token assignment produced by GridGroup.

        Args:
            group_feats: (B, G, C) token features.
            inverse:     (B, N) long, point -> token id in [0, G).

        Returns:
            (B, N, C) per-point features.
        """
        B, N = inverse.shape
        C = group_feats.shape[-1]
        return group_feats.gather(1, inverse.unsqueeze(-1).expand(-1, -1, C))  # (B,N,C)

    def forward(self, pts):
        B, N, C = pts.shape

        # ---- group + encode ----
        if self.use_grid_group:
            neighborhood, center, inverse = self.group_divider(pts)  # (B,G,M,C),(B,G,3),(B,N)
        else:
            neighborhood, center = self.group_divider(pts)  # (B,G,M,C), (B,G,3)
            inverse = None
        # rgb_film 时 encoder 额外返回 per-group color_token，供 Mamba 层间 FiLM 调制
        color_token = None
        if self.rgb_film:
            group_tokens, color_token = self.encoder(neighborhood, return_color=True)
        else:
            group_tokens = self.encoder(neighborhood)   # (B,G,encoder_dims)
        pos = self.pos_embed(center)                    # (B,G,trans_dim)

        # ---- Hilbert serialization (single direction; keep faithful to PointMamba) ----
        _, order, inverse_order, group_tokens, pos = serialization_func(
            center, group_tokens, pos, 'hilbert')
        group_tokens = apply_OrderScale(group_tokens, self.OrderScale_gamma_1, self.OrderScale_beta_1)
        if color_token is not None:
            # color_token 也按同一 hilbert order 重排，与 group_tokens 对齐
            color_token = color_token.flatten(0, 1)[order].reshape(B, -1, self.trans_dim).contiguous()

        # ---- Mamba blocks ----
        x = self.blocks(group_tokens, pos, color_tokens=color_token)  # (B,G,trans_dim) serialized

        # ---- un-serialize back to original center order ----
        x = x.flatten(0, 1)[inverse_order].reshape(B, -1, self.trans_dim).contiguous()

        # ---- upsample group tokens -> points ----
        if self.use_grid_group:
            up = self.grid_upsample(x, inverse)  # O(N) inverse-gather (B,N,trans_dim)
        else:
            up = self.upsample(x, center, pts)  # (B,N,trans_dim)

        # ---- skip connection: per-point low-level feature ----
        point_feat = self.point_branch(pts.transpose(1, 2))  # (B,128,N)
        feat = torch.cat([up.transpose(1, 2), point_feat], dim=1)  # (B,trans_dim+128,N)

        # ---- heads ----
        logits = self.seg_head(feat).transpose(1, 2)  # (B,N,num_classes)

        mat_logits = None
        if self.num_materials > 0:
            if self.cascade_material:
                # 级联：把语义概率(softmax, detach)拼到材质头输入，提供"这是什么物体"的先验。
                # detach 防止材质损失回传污染语义头（语义只由 nuScenes 训练）。
                sem_prob = torch.softmax(logits, dim=-1).detach().transpose(1, 2)  # (B,num_classes,N)
                mat_feat = torch.cat([feat, sem_prob], dim=1)  # (B, trans_dim+128+num_classes, N)
            else:
                mat_feat = feat
            mat_logits = self.material_head(mat_feat).transpose(1, 2)  # (B,N,num_materials)

        if self.num_instances > 0:
            inst = self.instance_head(feat).transpose(1, 2)  # (B,N,num_instances)
            return logits, mat_logits, inst

        return logits, mat_logits

    def get_loss(self, ret, gt):
        """Per-point cross-entropy loss.

        Args:
            ret: (B, N, num_classes) logits
            gt:  (B, N) point-wise semantic labels (-1 / ignore index is masked)
        """
        B, N, C = ret.shape
        ret = ret.reshape(-1, C)
        gt = gt.reshape(-1).long()
        # mask out invalid labels (< 0)
        valid = gt >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=ret.device, requires_grad=True)
        # Focal Loss 分支优先于类加权 CE（两者是独立的消融变量，不叠加）。
        if self.focal_gamma is not None and self.focal_gamma > 0:
            logp = F.log_softmax(ret[valid], dim=-1)
            ce = -logp.gather(1, gt[valid].unsqueeze(1)).squeeze(1)
            pt = torch.exp(-ce).clamp(min=1e-8)
            loss = ((1 - pt) ** self.focal_gamma * ce).mean()
            return loss
        crit = self.loss_ce
        if self.loss_ce_sem is not None and C == self.sem_class_weight.numel():
            crit = self.loss_ce_sem
        loss = crit(ret[valid], gt[valid])
        return loss

    def load_model_from_ckpt(self, ckpt_path):
        """从训练 checkpoint 载入权重（用于微调/续训初始化）。"""
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        base = ckpt.get('base_model') or ckpt.get('model')
        base = {k.replace('module.', ''): v for k, v in base.items()}
        self.load_state_dict(base, strict=False)
        print(f'[PointMambaSeg] loaded pretrained weights from {ckpt_path}')
