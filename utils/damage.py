"""
破坏模型：实例 → 子部位分解 + 每部位强度 + 撞击 → 局部破坏（位置相关）。

核心思路（更真实的损坏逻辑）：
  1. 每个物体分解为若干「子部位」（k-means 空间分区）。
  2. 每部位强度 = 材料强度 × 几何结构强度（薄/稀疏部位更弱）。
  3. 撞击只作用于撞击点附近的部位（局部化）——因此「撞不同位置，坏不同部位」。
  4. 部位损伤累积 > 阈值 → 该部位破碎；物体按破碎比例进入 intact/damaged/destroyed。
  5. 损毁响应（damage_response）区分：
       none     —— 刚性不可破坏（混凝土/岩石），撞击无碎块
       deform   —— 只变形不碎（金属车体），累积到 destroyed 但不掉碎块
       break    —— 局部断裂（塑料/木材），仅受撞部位破碎
       shatter  —— 整体碎裂（玻璃），任一部位破阈值即全碎

单位：能量用焦耳；部位 hp = strength × HP_SCALE（HP_SCALE=1000 J，可标定）。
"""

from typing import Dict, List, Optional
import numpy as np

from .materials import PHYSICAL_MATERIALS, CLASS_MASS, CLASS_STATIC

# 材料断裂强度（0~1，越高越难破坏；近似屈服强度）
MATERIAL_STRENGTH: Dict[str, float] = {
    'soft_tissue': 0.05,
    'metal':       0.90,
    'glass':       0.12,
    'plastic':     0.30,
    'rubber':      0.45,
    'wood':        0.40,
    'vegetation':  0.20,
    'soil':        0.15,
    'gravel':      0.10,
    'rock':        0.95,
    'concrete':    0.95,
    'asphalt':     0.90,
    'unknown':     0.50,
}

HP_SCALE = 1000.0          # 部位 hp = strength × HP_SCALE（焦耳）
DESTROY_FRACTION = 0.5     # 破碎部位比例超过此值 → destroyed


def _geometric_factor(xyz: np.ndarray) -> float:
    """几何结构强度：用点云协方差奇异值估「厚实度」。

    compactness = σ3/σ1：扁平/细长部位（薄板、树枝）→ 小 → 更弱；
    紧凑块（树干、车体）→ 大 → 更强。
    """
    if len(xyz) < 3:
        return 1.0
    c = xyz - xyz.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    if s[0] < 1e-6:
        return 1.0
    compactness = float(s[-1] / s[0])
    return float(np.clip(compactness / 0.15, 0.25, 1.0))


def decompose_parts(xyz: np.ndarray, k: Optional[int] = None) -> List[Dict]:
    """把实例点云分解为 k 个子部位，返回每部位的几何与强度信息。"""
    from sklearn.cluster import KMeans

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    if k is None:
        k = max(2, min(8, int(n // 200)))
    if n < 40:
        k = 1
    k = max(1, min(k, n))

    if k == 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        k = min(k, n)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(xyz)

    parts = []
    sizes = np.bincount(labels, minlength=k)
    for i in range(k):
        if sizes[i] == 0:
            continue  # KMeans 退化可能产生空簇
        m = labels == i
        sub = xyz[m]
        parts.append(dict(
            idx=i,
            points=sub,
            center=sub.mean(axis=0).astype(np.float32),
            bbox_min=sub.min(axis=0).astype(np.float32),
            bbox_max=sub.max(axis=0).astype(np.float32),
            n_points=int(sizes[i]),
            geometric_factor=_geometric_factor(sub),
        ))
    if not parts:  # 兜底：单部位
        parts = [dict(idx=0, points=xyz,
                      center=xyz.mean(axis=0).astype(np.float32),
                      bbox_min=xyz.min(axis=0).astype(np.float32),
                      bbox_max=xyz.max(axis=0).astype(np.float32),
                      n_points=n, geometric_factor=_geometric_factor(xyz))]
    return parts


class DamageModel:
    """单个实例的破坏状态机（位置相关局部破坏）。"""

    def __init__(self, xyz, class_id, physical: str, damage_response: str,
                 density: float = 1000.0):
        self.xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        self.class_id = class_id
        self.physical = physical
        self.damage_response = damage_response
        self.density = density

        strength = MATERIAL_STRENGTH.get(physical, 0.5)
        self.parts = decompose_parts(self.xyz)
        for p in self.parts:
            p['strength'] = strength * p['geometric_factor']
            p['hp'] = p['strength'] * HP_SCALE
            p['damage'] = 0.0
            p['state'] = 'intact'      # intact / broken
        self.state = 'intact'          # intact / damaged / destroyed

        # 静态判定 + 质量：优先每类典型质量，否则密度 × 凸包体积（fallback 包围盒）
        self.is_static = class_id in CLASS_STATIC
        if class_id in CLASS_MASS:
            self.mass = CLASS_MASS[class_id]
        else:
            self.mass = density * self._volume()
        self._impact_count = 0

    def _volume(self) -> float:
        """凸包体积（比包围盒更接近真实体积）；退化时回退包围盒。"""
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.xyz)
            return float(hull.volume)
        except Exception:
            bb = self.bbox()
            return max(float(np.prod(bb[1] - bb[0])), 1e-6)

    def bbox(self):
        return (self.xyz.min(axis=0), self.xyz.max(axis=0))

    @property
    def broken_fraction(self) -> float:
        if not self.parts:
            return 0.0
        return sum(1 for p in self.parts if p['state'] == 'broken') / len(self.parts)

    def _object_state(self) -> str:
        frac = self.broken_fraction
        if frac >= DESTROY_FRACTION:
            return 'destroyed'
        if frac > 0:
            return 'damaged'
        return 'intact'

    def apply_impact(self, point, energy: float, radius: float = 0.5) -> Dict:
        """在 point 处施加 energy 焦耳冲击，返回破坏事件。

        Returns:
            dict(state, broken_parts, debris, note)
        """
        point = np.asarray(point, dtype=np.float32).reshape(3)
        self._impact_count += 1

        # 不可破坏：无碎块
        if self.damage_response == 'none':
            self.state = self._object_state()
            return dict(state=self.state, broken_parts=[], debris=[], note='刚性不可破坏')

        # 找受影响部位：距撞击点 radius 内的部位；若没有则取最近部位
        centers = np.array([p['center'] for p in self.parts], dtype=np.float32)
        dists = np.linalg.norm(centers - point, axis=1)
        affected = list(np.where(dists <= radius)[0])
        if not affected:
            affected = [int(np.argmin(dists))]

        broken = []
        for pi in affected:
            p = self.parts[pi]
            if p['state'] == 'broken':
                continue
            p['damage'] += energy / max(p['hp'], 1e-6)

            # 玻璃 shatter：任一部位破阈值 → 全部碎
            if self.damage_response == 'shatter' and p['damage'] >= 1.0:
                for q in self.parts:
                    q['state'] = 'broken'
                broken = [q['idx'] for q in self.parts]
                break

            if p['damage'] >= 1.0:
                p['state'] = 'broken'
                broken.append(pi)

        self.state = self._object_state()

        # 掉落的碎块（deform 不掉碎块，break/shatter 掉）
        debris = []
        if self.damage_response in ('break', 'shatter'):
            debris = [self.parts[i] for i in broken]

        note = (f'{self.damage_response}: {len(broken)}/{len(self.parts)} 部位破碎'
                if broken else f'{self.damage_response}: 无破碎（能量不足）')
        return dict(state=self.state, broken_parts=broken, debris=debris, note=note)

    def summary(self) -> Dict:
        return dict(
            class_id=self.class_id,
            physical=self.physical,
            damage_response=self.damage_response,
            is_static=self.is_static,
            mass=self.mass,
            state=self.state,
            broken_fraction=self.broken_fraction,
            parts=[dict(idx=p['idx'], center=p['center'].tolist(),
                        strength=round(p['strength'], 4), hp=round(p['hp'], 1),
                        damage=round(p['damage'], 3), state=p['state'],
                        geometric_factor=round(p['geometric_factor'], 3))
                   for p in self.parts],
        )
