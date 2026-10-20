"""GridSampling + voxel-patch grouping (方案 B).

Replaces ``Group`` (FPS + KNN) with voxel grid sampling to remove the
``O(B*G*N)`` distance-matrix memory bottleneck of the KNN grouping, so the
token count (``num_group``) and the input point count (``npoints``) can scale
to tens of thousands of points per frame.

Each occupied voxel = one token ("patch"); the points inside that voxel are its
members.  The point -> token assignment is deterministic (an ``inverse`` mapping),
which also yields an ``O(N)`` upsampling path (no KNN on the decode side).

Compared to ``Group`` (FPS + KNN, dense BGM):
    * grouping   : O(N)      voxel hashing  (was O(B*G*N) KNN distance matrix)
    * upsampling : O(N)      inverse gather (was O(B*N*G) KNN distance matrix)
"""
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils


class GridGroup(nn.Module):
    """Voxel-grid token formation with deterministic point -> token mapping.

    Args:
        num_group:  target token count (upper bound; padded with dummy tokens if
                    fewer voxels are occupied, FPS-subsampled if more).
        group_size: max member points per voxel (patch size).
        grid_size:  voxel edge length (meters).
    """

    def __init__(self, num_group, group_size, grid_size=0.5):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.grid_size = grid_size

    def forward(self, xyz):
        """Group points into voxel tokens.

        Args:
            xyz: (B, N, C); first 3 channels are XYZ, the rest (e.g. RGB) are
                 carried through as patch members.

        Returns:
            neighborhood: (B, num_group, group_size, C) members per token, XYZ
                          normalized relative to the token center.
            center:       (B, num_group, 3) token (voxel) centers.
            inverse:      (B, N) long, point -> token id in [0, num_group).
        """
        B, N, C = xyz.shape
        device = xyz.device
        coords = xyz[:, :, :3]   # (B, N, 3)
        extra = xyz[:, :, 3:]    # (B, N, C-3)

        neigh_list, center_list, inverse_list = [], [], []
        for b in range(B):
            nb, ce, inv = self._group_single(coords[b], extra[b], device)
            neigh_list.append(nb)
            center_list.append(ce)
            inverse_list.append(inv)

        neighborhood = torch.stack(neigh_list, dim=0)   # (B, num_group, M, C)
        center = torch.stack(center_list, dim=0)        # (B, num_group, 3)
        inverse = torch.stack(inverse_list, dim=0)      # (B, N)
        return neighborhood, center, inverse

    def _group_single(self, coords, extra, device):
        """Group one batch sample. coords (N,3), extra (N,C-3)."""
        N = coords.shape[0]
        C_minus = extra.shape[1]
        C = 3 + C_minus

        # 1) voxelize (O(N) hashing)
        grid = torch.floor(coords / self.grid_size).to(torch.long)  # (N,3)
        min_g = grid.min(dim=0, keepdim=True).values                 # (1,3)
        grid_shift = grid - min_g                                    # (N,3) >= 0
        size = grid_shift.max(dim=0).values + 1                      # (3,)
        lin = (grid_shift[:, 0] * (size[1] * size[2])
               + grid_shift[:, 1] * size[2]
               + grid_shift[:, 2])                                   # (N,)
        uniq, inv = torch.unique(lin, return_inverse=True)           # (G_b,), (N,)
        G_b = uniq.shape[0]

        # recover voxel grid coords from linear index
        gz = uniq % size[2]
        gy = (uniq // size[2]) % size[1]
        gx = uniq // (size[1] * size[2])
        cell_grid = torch.stack([gx, gy, gz], dim=-1) + min_g        # (G_b,3)
        center_all = (cell_grid.float() + 0.5) * self.grid_size      # (G_b,3)

        # 2) cap token count to num_group (FPS on voxel centers if too many)
        if G_b > self.num_group:
            sel = pointnet2_utils.furthest_point_sample(
                center_all.float().contiguous().unsqueeze(0),
                self.num_group,
            ).squeeze(0).long()                                       # (num_group,)
            center_use = center_all[sel]                              # (num_group,3)

            remap = torch.full((G_b,), -1, dtype=torch.long, device=device)
            remap[sel] = torch.arange(self.num_group, device=device)
            # dropped voxels -> nearest kept voxel (voxel-level, cheap)
            dropped_idx = torch.nonzero(remap < 0).squeeze(1)         # (G_dropped,)
            if dropped_idx.numel() > 0:
                d_centers = center_all[dropped_idx]                   # (D,3)
                dist = torch.cdist(d_centers, center_use)             # (D, num_group)
                remap[dropped_idx] = dist.argmin(dim=1)
            inv_token = remap[inv]                                    # (N,)
            G_use = self.num_group
        else:
            center_use = center_all
            inv_token = inv
            G_use = G_b

        # 3) gather up to group_size members per token (sort + rank + scatter)
        order = torch.argsort(inv_token, stable=True)
        inv_sorted = inv_token[order]                                 # (N,)
        counts = torch.bincount(inv_sorted, minlength=G_use)          # (G_use,)
        start = torch.cumsum(counts, dim=0) - counts                  # (G_use,)
        rank = torch.arange(N, device=device) - start[inv_sorted]     # (N,)
        keep = rank < self.group_size                                 # (N,)
        sel_sorted = torch.nonzero(keep).squeeze(1)                   # (n_keep,)
        sel_token = inv_sorted[sel_sorted]                            # (n_keep,)
        sel_rank = rank[sel_sorted]                                   # (n_keep,)
        sel_point = order[sel_sorted]                                 # (n_keep,)
        flat_slot = sel_token * self.group_size + sel_rank

        # pad with token center (XYZ offset 0) and zero extra channels
        neigh = center_use.unsqueeze(1).expand(G_use, self.group_size, 3).clone()
        neigh.view(-1, 3)[flat_slot] = coords[sel_point]
        neigh = neigh - center_use.unsqueeze(1)                       # rel XYZ

        if C_minus > 0:
            neigh_extra = torch.zeros(G_use, self.group_size, C_minus, device=device)
            neigh_extra.view(-1, C_minus)[flat_slot] = extra[sel_point]
            neighborhood = torch.cat([neigh, neigh_extra], dim=-1)    # (G,M,C)
        else:
            neighborhood = neigh                                      # (G,M,3)

        # 4) pad to num_group with dummy tokens (center 0, no members reference them)
        if G_use < self.num_group:
            pad = self.num_group - G_use
            neighborhood = torch.cat([
                neighborhood,
                torch.zeros(pad, self.group_size, C, device=device),
            ], dim=0)
            center_use = torch.cat([
                center_use,
                torch.zeros(pad, 3, device=device),
            ], dim=0)

        return neighborhood, center_use, inv_token
