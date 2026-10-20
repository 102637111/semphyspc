#!/usr/bin/env python3
"""方案 B 冒烟测试：GridGroup 正确性 + PointMambaSeg(grid) 前向/反向。

用法: python scripts/test_grid_group.py
"""
import types
import torch

from models.grid_group import GridGroup
from models.point_mamba_seg import PointMambaSeg


def make_config(**over):
    base = dict(
        trans_dim=384, depth=12, group_size=32, num_group=1024,
        encoder_dims=384, in_channels=6, num_classes=32, upsample_k=3,
        rms_norm=False, drop_path=0.0, drop_out=0.0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_grid_group_correctness():
    # --- controlled small case: no FPS downsampling, no truncation ---
    grid_size = 1.0
    coords = torch.tensor([
        [0.1, 0.2, 0.3],   # voxel (0,0,0)
        [0.6, 0.7, 0.8],   # voxel (0,0,0)
        [1.1, 1.2, 1.3],   # voxel (1,1,1)
        [2.9, 0.1, 0.1],   # voxel (2,0,0)
        [-0.1, -0.2, -0.3],# voxel (-1,-1,-1)
    ], dtype=torch.float32).cuda()
    rgb = torch.zeros(5, 3).cuda()
    pts = torch.cat([coords, rgb], dim=-1).unsqueeze(0)  # (1,5,6)

    gg = GridGroup(num_group=100, group_size=8, grid_size=grid_size)
    neighborhood, center, inverse = gg(pts)

    # expected voxels: (0,0,0), (1,1,1), (2,0,0), (-1,-1,-1) -> 4 tokens
    # token order is arbitrary (torch.unique order), so compare center set + inverse equivalence
    centers = center[0]  # (100,3)
    real = centers[centers.abs().sum(-1) > 0]  # (4,3) non-padded
    assert real.shape[0] == 4, f'expected 4 voxels, got {real.shape[0]}'

    def voxel_of(c):
        return tuple((torch.floor(c / grid_size) + 0.5 * grid_size).tolist())

    # every point's token center must equal its own voxel center
    for n in range(5):
        t = inverse[0, n].item()
        expected_center = torch.tensor(voxel_of(coords[n])).cuda()
        assert torch.allclose(center[0, t], expected_center, atol=1e-4), \
            f'point {n} -> token {t} center {center[0,t]} != {expected_center}'

    # two points (0,1) share voxel (0,0,0): same token
    assert inverse[0, 0].item() == inverse[0, 1].item()

    # rel-offset round trip for each point (all placed: group_size=8 > voxel size 2)
    for n in range(5):
        t = inverse[0, n].item()
        rel = neighborhood[0, t, :, :3]
        want = coords[n] - center[0, t]
        # the point must be present in one of the member slots
        assert torch.any(torch.all(torch.isclose(rel, want, atol=1e-4), dim=-1)), \
            f'point {n} not found among token {t} members'
    print('[GridGroup] controlled small case: inverse + centers + round-trip OK')

    # --- large random case: triggers FPS downsampling (16384 pts / 0.5 grid >> 1024) ---
    torch.manual_seed(0)
    B, N, C = 2, 16384, 6
    big = torch.randn(B, N, C).cuda()
    big[:, :, 3:] = torch.rand(B, N, 3).cuda()
    gg2 = GridGroup(num_group=1024, group_size=32, grid_size=0.5)
    neighborhood, center, inverse = gg2(big)
    print(f'[GridGroup] large: neighborhood {tuple(neighborhood.shape)}, '
          f'center {tuple(center.shape)}, inverse {tuple(inverse.shape)}')
    assert inverse.min() >= 0 and inverse.max() < 1024
    # non-padded member slots must map back to real coordinates
    rel = neighborhood[:, :, :, :3]                     # (B,G,M,3)
    abs_xyz = rel + center.unsqueeze(2)                 # (B,G,M,3)
    nz = rel.abs().sum(-1) > 0                          # placed slots (non-zero offset)
    # every placed slot's absolute coords must be an exact copy of some input point
    placed = abs_xyz[nz]                                # (K,3)
    assert placed.shape[0] > 0
    # sample to keep memory bounded; use manual broadcast distance (torch.cdist
    # on this GPU is imprecise for near-identical points, returns ~1e-3).
    sample = placed[torch.randperm(placed.shape[0])[:500]]
    all_coords = big[:, :, :3].reshape(-1, 3)            # (B*N, 3)
    d = ((sample.unsqueeze(1) - all_coords.unsqueeze(0)) ** 2).sum(-1).sqrt()
    min_d = d.min(dim=1).values
    assert min_d.max() < 1e-4, f'max min-dist {min_d.max():.4f} too large'
    print('[GridGroup] large random case OK (FPS path exercised)')


def test_model_forward(use_grid_group, tag):
    torch.manual_seed(0)
    B, N, C = 2, 16384, 6
    cfg = make_config(use_grid_group=use_grid_group, grid_size=0.5)
    model = PointMambaSeg(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{tag}] params = {n_params/1e6:.2f}M')

    pts = torch.randn(B, N, C).cuda()
    pts[:, :, 3:] = torch.rand(B, N, 3).cuda()
    sem = torch.randint(0, 32, (B, N)).cuda()
    sem[0, -100:] = -1  # some ignore labels

    logits, mat_logits = model(pts)
    print(f'[{tag}] logits {tuple(logits.shape)}, mat_logits {mat_logits}')
    assert tuple(logits.shape) == (B, N, 32)
    assert mat_logits is None

    loss = model.get_loss(logits, sem)
    print(f'[{tag}] loss = {loss.item():.4f}')
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f'[{tag}] backward OK, |grad| sum = {g:.2f}')
    del model, pts, sem, logits
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_grid_group_correctness()
    print('=' * 60)
    test_model_forward(use_grid_group=True, tag='grid-group')
    print('=' * 60)
    test_model_forward(use_grid_group=False, tag='fps-knn   ')
    print('\n全部通过 ✓')
