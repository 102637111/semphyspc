"""
实例级聚类：把逐点语义标签 + 坐标聚成独立的物体实例。

thing / stuff 划分：路面/人行道/地形等连续「背景」不聚类为实例；
其余可移动物体、车辆、行人、植被、建筑等聚成独立实例。

用法：
    from utils.instances import cluster_instances
    inst_id, insts = cluster_instances(sem, xyz)
    # inst_id: (N,) int32，0=背景/未聚类，>=1 为实例编号
    # insts:   dict instance_id -> summary（class_id/center/bbox/n_points/...）
"""

from typing import Dict, List, Tuple
import numpy as np

from .materials import LIDARSEG_32, class_physics, class_meta

# 背景类别（不聚类）：noise + 路面/人行道/其他平坦面/地形
STUFF_CLASSES = {0, 24, 25, 26, 27}

# 每类 DBSCAN 半径 eps（米）。关键：能分开相邻的独立物体。
CLASS_EPS: Dict[int, float] = {
    1: 0.5,     # animal
    2: 0.4, 3: 0.4, 4: 0.4, 5: 0.5, 6: 0.4, 7: 0.4, 8: 0.4,   # pedestrians
    9: 0.5,     # barrier
    10: 0.5,    # debris
    11: 0.5,    # pushable_pullable
    12: 0.4,    # trafficcone
    13: 0.8,    # bicycle_rack
    14: 0.6,    # bicycle
    15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0,       # bus/car/construction/emergency
    21: 0.6,    # motorcycle
    22: 1.0,    # trailer
    23: 1.0,    # truck
    28: 2.0,    # manmade（建筑）
    29: 1.0,    # static.other
    30: 1.2,    # vegetation（树/灌木）
}


def cluster_instances(sem: np.ndarray, xyz: np.ndarray,
                      min_points: int = 8,
                      eps_override: Dict[int, float] = None) -> Tuple[np.ndarray, Dict]:
    """逐类 DBSCAN 聚类，返回 (instance_id, instances)。

    instance_id: (N,) int32，0=背景/噪声点，>=1 为实例编号（全局唯一，跨类递增）。
    instances:   dict instance_id -> dict(class_id, class_name, center, bbox_min,
                  bbox_max, n_points, physical, safety, lethality)
    """
    from sklearn.cluster import DBSCAN

    sem = np.asarray(sem).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    eps_map = dict(CLASS_EPS)
    if eps_override:
        eps_map.update(eps_override)

    instance_id = np.zeros(len(sem), dtype=np.int32)
    instances: Dict[int, Dict] = {}
    next_id = 1

    for cid in sorted(set(int(c) for c in np.unique(sem))):
        if cid in STUFF_CLASSES:
            continue
        mask = sem == cid
        pts = xyz[mask]
        if len(pts) < min_points:
            continue
        eps = eps_map.get(cid, 1.0)
        labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(pts)
        for lb in sorted(set(labels)):
            if lb < 0:
                continue  # DBSCAN 噪声 → 视为背景
            m = labels == lb
            n = int(m.sum())
            if n < min_points:
                continue
            sub = pts[m]
            center = sub.mean(axis=0)
            bbox_min = sub.min(axis=0)
            bbox_max = sub.max(axis=0)
            iid = next_id
            next_id += 1
            # 注意：instance_id[mask][m] 是副本赋值，不会写回；须用整型索引
            full_idx = np.where(mask)[0]
            instance_id[full_idx[m]] = iid
            meta = class_meta(cid)
            ph = class_physics(cid)
            instances[iid] = dict(
                class_id=cid,
                class_name=LIDARSEG_32[cid],
                center=center.astype(np.float32),
                bbox_min=bbox_min.astype(np.float32),
                bbox_max=bbox_max.astype(np.float32),
                n_points=n,
                physical=ph['physical'],
                safety=ph['safety'],
                lethality=ph['lethality'],
                destructibility=ph['destructibility'],
                density=ph['density'],
                friction=ph['friction'],
                restitution=ph['restitution'],
                damage_response=meta['damage_response'],
            )
    return instance_id, instances
