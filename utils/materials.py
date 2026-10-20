"""
物理材质库 + 语义类别 → 物理交互属性映射。

研究目标（与应用直接对应）：
  1. 导航失控安全碰撞 —— 失控时主动选择撞击「非致命、可变形/可破坏」的物体
     （灌木/交通锥/塑料护栏）或刮蹭护栏来减速，而不是撞硬物（车/墙）或行人/动物。
  2. 游戏建图 —— 实地扫描的点云地图本身没有碰撞/损毁逻辑，按类别赋
     「碰撞 + 损毁」参数（质量/摩擦/恢复系数/可破坏性/损伤响应），
     使游戏建模更简单、更真实。

因此这里的「材质」核心是物理交互属性，视觉 PBR 仅作次要的渲染属性。

每个语义类 → 一份 CLASS_META，含：
  physical       : 物理材质名（查 PHYSICAL_MATERIALS 得密度/摩擦/恢复/刚性/可破坏性）
  visual         : 视觉材质名（查 VISUAL_MATERIALS 得 PBR 底色/金属度/粗糙度）
  safety         : 碰撞安全等级（见 COLLISION_SAFETY）
  lethality      : 撞击代价 0~1（导航决策用，越高越不能撞）
  damage_response: 损毁响应（游戏损毁逻辑：none/deform/break/shatter）
  note           : 说明
"""

from typing import Dict, List, Optional

# nuScenes-lidarseg 32 类（索引顺序，与 scripts/eval_seg_full.py 一致）
LIDARSEG_32 = [
    'noise', 'animal',
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer', 'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier', 'movable_object.debris',
    'movable_object.pushable_pullable', 'movable_object.trafficcone',
    'static_object.bicycle_rack', 'vehicle.bicycle',
    'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
    'vehicle.construction', 'vehicle.emergency.ambulance',
    'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
    'vehicle.truck', 'flat.driveable_surface', 'flat.other',
    'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other',
    'static.vegetation', 'vehicle.ego',
]

# ---------------- 碰撞安全等级 ----------------
# 导航决策用：从「绝对不能撞」到「正常可通过」。
COLLISION_SAFETY = ['lethal', 'hard', 'frangible', 'deformable', 'passable']

COLLISION_SAFETY_CN = {
    'lethal':    '致命（行人/动物，严禁撞击）',
    'hard':      '硬刚性（车/墙/建筑，撞击严重受损，尽量避让）',
    'frangible': '可破坏（护栏/交通锥/玻璃，可主动撞击减速）',
    'deformable':'可变形（灌木/软土/草，可安全穿越或刮蹭减速）',
    'passable':  '可通过（路面/地形，正常行驶）',
}

# 每个安全等级对应的默认撞击代价（lethality）
SAFETY_LETHALITY_DEFAULT = {'lethal': 1.0, 'hard': 0.85,
                            'frangible': 0.3, 'deformable': 0.15, 'passable': 0.0}

# ---------------- 物理材质库（密度/摩擦/恢复/刚性/可破坏性） ----------------
# density: kg/m^3（质量代理，物理引擎 Rigidbody mass 用）
# friction: 滑动摩擦系数（物理引擎 PhysicMaterial 用）
# restitution: 恢复系数 bounciness 0~1
# rigidity: rigid | semi_rigid | deformable | frangible | fluid
# destructibility: 0~1 可破坏程度（游戏损毁/血条阈值用）
PHYSICAL_MATERIALS: Dict[str, Dict] = {
    'soft_tissue':   dict(density=1000, friction=0.50, restitution=0.00,
                          rigidity='deformable', destructibility=0.90,
                          desc='人体/动物软组织'),
    'metal':         dict(density=7850, friction=0.50, restitution=0.10,
                          rigidity='rigid', destructibility=0.05,
                          desc='刚性金属（钢/铝）'),
    'glass':         dict(density=2500, friction=0.40, restitution=0.05,
                          rigidity='frangible', destructibility=0.95,
                          desc='玻璃（碎）'),
    'plastic':       dict(density=1000, friction=0.40, restitution=0.40,
                          rigidity='frangible', destructibility=0.50,
                          desc='塑料（可变形/可碎）'),
    'rubber':        dict(density=1100, friction=0.90, restitution=0.70,
                          rigidity='deformable', destructibility=0.30,
                          desc='橡胶（轮胎/锥桶）'),
    'wood':          dict(density=700, friction=0.50, restitution=0.20,
                          rigidity='frangible', destructibility=0.40,
                          desc='木材（可折断）'),
    'vegetation':    dict(density=300, friction=0.50, restitution=0.05,
                          rigidity='deformable', destructibility=0.80,
                          desc='植被/灌木/草（可穿越）'),
    'soil':          dict(density=1500, friction=0.60, restitution=0.00,
                          rigidity='deformable', destructibility=0.00,
                          desc='土壤/泥地'),
    'gravel':        dict(density=1800, friction=0.60, restitution=0.00,
                          rigidity='deformable', destructibility=0.00,
                          desc='碎石/沙砾'),
    'rock':          dict(density=2600, friction=0.70, restitution=0.05,
                          rigidity='rigid', destructibility=0.00,
                          desc='岩石/岩体（硬）'),
    'concrete':      dict(density=2400, friction=0.70, restitution=0.05,
                          rigidity='rigid', destructibility=0.00,
                          desc='混凝土（墙/建筑/人行道）'),
    'asphalt':       dict(density=2300, friction=0.80, restitution=0.05,
                          rigidity='rigid', destructibility=0.00,
                          desc='沥青（路面）'),
    'unknown':       dict(density=1000, friction=0.50, restitution=0.10,
                          rigidity='rigid', destructibility=0.00,
                          desc='未知/兜底'),
}

# ---------------- 视觉材质库（PBR，次要：渲染用） ----------------
VISUAL_MATERIALS: Dict[str, Dict] = {
    'asphalt':      dict(base_color=(0.12, 0.12, 0.12), metallic=0.00, roughness=0.90),
    'concrete':     dict(base_color=(0.55, 0.55, 0.55), metallic=0.00, roughness=0.85),
    'soil':         dict(base_color=(0.45, 0.35, 0.25), metallic=0.00, roughness=1.00),
    'gravel':       dict(base_color=(0.50, 0.48, 0.45), metallic=0.00, roughness=1.00),
    'rock':         dict(base_color=(0.40, 0.38, 0.36), metallic=0.00, roughness=0.95),
    'vegetation':   dict(base_color=(0.22, 0.45, 0.15), metallic=0.00, roughness=0.90),
    'wood':         dict(base_color=(0.50, 0.35, 0.20), metallic=0.00, roughness=0.70),
    'metal':        dict(base_color=(0.70, 0.70, 0.72), metallic=1.00, roughness=0.40),
    'painted_metal': dict(base_color=(0.60, 0.60, 0.65), metallic=0.70, roughness=0.35),
    'glass':        dict(base_color=(0.85, 0.90, 0.95), metallic=0.00, roughness=0.05),
    'plastic':      dict(base_color=(0.90, 0.50, 0.10), metallic=0.00, roughness=0.50),
    'rubber':       dict(base_color=(0.05, 0.05, 0.05), metallic=0.00, roughness=0.95),
    'cloth':        dict(base_color=(0.40, 0.40, 0.45), metallic=0.00, roughness=0.90),
    'fur':          dict(base_color=(0.60, 0.45, 0.35), metallic=0.00, roughness=0.80),
    'skin':         dict(base_color=(0.62, 0.45, 0.38), metallic=0.00, roughness=0.75),
    'unknown':      dict(base_color=(0.30, 0.30, 0.30), metallic=0.00, roughness=0.90),
}

# ---------------- 类别 → 属性（32 类，索引对齐 LIDARSEG_32） ----------------
# physical / visual / safety / lethality / damage_response
CLASS_META: List[Dict] = [
    # 0 noise
    dict(physical='unknown',    visual='unknown',    safety='passable',   lethality=0.00, damage_response='none',    note='噪声/无效点'),
    # 1 animal
    dict(physical='soft_tissue', visual='fur',        safety='lethal',    lethality=1.00, damage_response='deform',  note='动物（严禁撞击）'),
    # 2-8 行人
    dict(physical='soft_tissue', visual='cloth',      safety='lethal',    lethality=1.00, damage_response='deform',  note='行人-成人'),
    dict(physical='soft_tissue', visual='cloth',      safety='lethal',    lethality=1.00, damage_response='deform',  note='行人-儿童'),
    dict(physical='soft_tissue', visual='cloth',      safety='lethal',    lethality=1.00, damage_response='deform',  note='建筑工人'),
    dict(physical='metal',       visual='metal',      safety='hard',      lethality=0.80, damage_response='deform',  note='代步车'),
    dict(physical='soft_tissue', visual='cloth',      safety='lethal',    lethality=1.00, damage_response='deform',  note='警察'),
    dict(physical='soft_tissue', visual='cloth',      safety='lethal',    lethality=1.00, damage_response='deform',  note='婴儿车'),
    dict(physical='soft_tissue', visual='cloth',      safety='lethal',    lethality=1.00, damage_response='deform',  note='轮椅'),
    # 9 movable_object.barrier 护栏
    dict(physical='plastic',     visual='plastic',    safety='frangible', lethality=0.30, damage_response='deform',  note='护栏/路障（可刮蹭减速）'),
    # 10 debris
    dict(physical='gravel',      visual='gravel',     safety='deformable',lethality=0.15, damage_response='none',    note='碎屑/砂石'),
    # 11 pushable_pullable
    dict(physical='plastic',     visual='plastic',    safety='frangible', lethality=0.25, damage_response='break',   note='推拉物/行李箱'),
    # 12 trafficcone
    dict(physical='rubber',      visual='plastic',    safety='frangible', lethality=0.10, damage_response='deform',  note='交通锥（可撞击）'),
    # 13 bicycle_rack
    dict(physical='metal',       visual='metal',      safety='hard',      lethality=0.75, damage_response='deform',  note='自行车架'),
    # 14 bicycle
    dict(physical='metal',       visual='metal',      safety='hard',      lethality=0.75, damage_response='deform',  note='自行车'),
    # 15/16 bus
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='铰接公交'),
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='公交'),
    # 17 car
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='轿车'),
    # 18 construction
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='工程车'),
    # 19/20 emergency
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='救护车'),
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='警车'),
    # 21 motorcycle
    dict(physical='metal',       visual='metal',      safety='hard',      lethality=0.80, damage_response='deform',  note='摩托车'),
    # 22 trailer
    dict(physical='metal',       visual='metal',      safety='hard',      lethality=0.80, damage_response='deform',  note='拖车'),
    # 23 truck
    dict(physical='metal',       visual='painted_metal', safety='hard',   lethality=0.90, damage_response='deform',  note='卡车'),
    # 24 driveable_surface
    dict(physical='asphalt',     visual='asphalt',    safety='passable',  lethality=0.00, damage_response='none',    note='可行驶路面'),
    # 25 flat.other
    dict(physical='concrete',    visual='concrete',   safety='passable',  lethality=0.10, damage_response='none',    note='其他平坦面'),
    # 26 sidewalk
    dict(physical='concrete',    visual='concrete',   safety='passable',  lethality=0.20, damage_response='none',    note='人行道（必要时可上）'),
    # 27 terrain
    # safety=passable（2026-09-14 由 deformable 改回）：地形是**行驶面**，不是障碍物。
    # 原分类让它进了 select_soft_contact 的候选集（该集只排除 passable），
    # 且 lethality 最低，导致 73% 的帧推荐「撞地面」。论文 §3.3 本就把它归为
    # 「路面/地形，正常行驶」，此处使代码与论文一致。
    dict(physical='soil',        visual='soil',       safety='passable',  lethality=0.05, damage_response='deform',  note='地形/泥地'),
    # 28 static.manmade
    dict(physical='concrete',    visual='concrete',   safety='hard',      lethality=0.90, damage_response='none',    note='建筑/墙体'),
    # 29 static.other
    dict(physical='rock',        visual='rock',       safety='hard',      lethality=0.70, damage_response='none',    note='其他静态物/岩石'),
    # 30 static.vegetation
    dict(physical='vegetation',  visual='vegetation', safety='deformable',lethality=0.20, damage_response='break',   note='植被/灌木（可穿越减速，树枝可断）'),
    # 31 vehicle.ego
    dict(physical='metal',       visual='metal',      safety='hard',      lethality=0.90, damage_response='deform',  note='自车'),
]

# 个别类视觉底色覆盖（交通锥橙、应急车白、工程黄、植被绿）
CLASS_COLOR_OVERRIDES: Dict[int, tuple] = {
    12: (0.95, 0.40, 0.05),   # trafficcone 橙色
    19: (0.95, 0.95, 0.95),   # ambulance 白色
    20: (0.95, 0.95, 0.95),   # emergency.police 白色
    18: (0.85, 0.65, 0.05),   # construction 工程黄
    30: (0.18, 0.40, 0.12),   # vegetation 绿色
}

# 每类典型质量（kg）。物理引擎 Rigidbody mass 用；优先于此，避免「密度×包围盒」高估。
CLASS_MASS: Dict[int, float] = {
    1: 50.0,      # animal
    2: 70.0, 3: 25.0, 4: 75.0, 5: 60.0, 6: 75.0, 7: 70.0, 8: 60.0,   # pedestrians
    9: 30.0,      # barrier
    10: 5.0,      # debris
    11: 20.0,     # pushable_pullable
    12: 5.0,      # trafficcone
    13: 30.0,     # bicycle_rack
    14: 15.0,     # bicycle
    15: 12000.0,  # bus.bendy
    16: 11000.0,  # bus.rigid
    17: 1500.0,   # car
    18: 15000.0,  # construction
    19: 2500.0,   # ambulance
    20: 2200.0,   # police
    21: 200.0,    # motorcycle
    22: 4000.0,   # trailer
    23: 8000.0,   # truck
    29: 500.0,    # static.other（岩石）
    30: 200.0,    # vegetation（树/灌木）
    31: 1500.0,   # ego
}

# 静态类：物理引擎中作为 kinematic/static 碰撞体（不参与动力学，质量视为无穷）
CLASS_STATIC: set = {0, 13, 24, 25, 26, 27, 28, 29, 30}

# 唯一 id
PHYSICAL_IDS: Dict[str, int] = {name: i for i, name in enumerate(PHYSICAL_MATERIALS.keys())}
ID_TO_PHYSICAL: Dict[int, str] = {i: name for name, i in PHYSICAL_IDS.items()}
SAFETY_IDS: Dict[str, int] = {s: i for i, s in enumerate(COLLISION_SAFETY)}


# ---------------- 查询接口 ----------------
def class_meta(class_id: int) -> Dict:
    if not (0 <= class_id < len(CLASS_META)):
        return CLASS_META[0]
    return CLASS_META[class_id]


def class_physics(class_id: int) -> Dict:
    """返回某类的物理参数（含物理材质库字段 + 安全等级 + 致命性）。"""
    meta = class_meta(class_id)
    p = dict(PHYSICAL_MATERIALS[meta['physical']])
    p['physical'] = meta['physical']
    p['safety'] = meta['safety']
    p['lethality'] = meta['lethality']
    p['damage_response'] = meta['damage_response']
    p['note'] = meta['note']
    return p


def class_visual(class_id: int) -> Dict:
    meta = class_meta(class_id)
    v = dict(VISUAL_MATERIALS[meta['visual']])
    if class_id in CLASS_COLOR_OVERRIDES:
        v['base_color'] = CLASS_COLOR_OVERRIDES[class_id]
    return v


# ---------------- 应用（逐点） ----------------
def apply_materials(sem_labels: "np.ndarray") -> Dict[str, "np.ndarray"]:
    """把逐点语义标签 (N,) 映射为逐点【物理 + 视觉】属性。

    Returns:
        physical_id : (N,) int32  物理材质 id（反查 PHYSICAL_MATERIALS / ID_TO_PHYSICAL）
        safety_id   : (N,) int32  碰撞安全等级 id（反查 COLLISION_SAFETY）
        lethality   : (N,) float32 撞击代价 0~1
        destructibility: (N,) float32 可破坏程度 0~1
        friction    : (N,) float32 摩擦系数
        density     : (N,) float32 密度 kg/m^3
        restitution : (N,) float32 恢复系数
        base_color  : (N,3) float32 视觉底色 RGB 0~1
        metallic    : (N,) float32 金属度
        roughness   : (N,) float32 粗糙度
    """
    import numpy as np
    sem = np.asarray(sem_labels).reshape(-1)
    n = sem.shape[0]

    physical_id = np.zeros(n, dtype=np.int32)
    safety_id = np.zeros(n, dtype=np.int32)
    lethality = np.zeros(n, dtype=np.float32)
    destructibility = np.zeros(n, dtype=np.float32)
    friction = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    restitution = np.zeros(n, dtype=np.float32)
    base_color = np.zeros((n, 3), dtype=np.float32)
    metallic = np.zeros(n, dtype=np.float32)
    roughness = np.zeros(n, dtype=np.float32)

    for cid in range(len(CLASS_META)):
        m = sem == cid
        if not m.any():
            continue
        ph = class_physics(cid)
        vi = class_visual(cid)
        physical_id[m] = PHYSICAL_IDS[ph['physical']]
        safety_id[m] = SAFETY_IDS[ph['safety']]
        lethality[m] = ph['lethality']
        destructibility[m] = ph['destructibility']
        friction[m] = ph['friction']
        density[m] = ph['density']
        restitution[m] = ph['restitution']
        base_color[m] = np.array(vi['base_color'], dtype=np.float32)
        metallic[m] = vi['metallic']
        roughness[m] = vi['roughness']

    invalid = (sem < 0) | (sem >= len(CLASS_META))
    if invalid.any():
        physical_id[invalid] = PHYSICAL_IDS['unknown']
        safety_id[invalid] = SAFETY_IDS['passable']
        base_color[invalid] = np.array(VISUAL_MATERIALS['unknown']['base_color'], dtype=np.float32)

    return dict(physical_id=physical_id, safety_id=safety_id, lethality=lethality,
                destructibility=destructibility, friction=friction, density=density,
                restitution=restitution, base_color=base_color, metallic=metallic,
                roughness=roughness)


# ---------------- 导航应用：失控安全碰撞目标选择 ----------------
def select_soft_contact(sem_labels, xyz, ego_xy=(0.0, 0.0), heading=0.0,
                        max_range=50.0, half_fov_deg=40.0, topk=5):
    """在前向视野内选择「最安全可撞击」的目标（导航失控软碰撞决策）。

    Args:
        sem_labels: (N,) 逐点语义标签
        xyz:        (N,3) 点坐标（ego 局部坐标系，x 前向 / y 左向）
        ego_xy:     ego 位置（通常 (0,0)）
        heading:    航向（弧度），0 = x 轴正前方
        max_range:  前向搜索距离
        half_fov_deg: 前向视锥半角（度）
        topk:       返回候选数

    Returns:
        dict: {ego_heading, candidates: [{class_name, class_id, safety, lethality,
               count, mean_dist, mean_xy}], recommended: <candidate>}
        按 lethality 升序（越安全越靠前），recommended 为最低致命性的可撞击候选。
    """
    import numpy as np
    sem = np.asarray(sem_labels).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

    # 前向视锥筛选（相对 ego）
    dx = xyz[:, 0] - ego_xy[0]
    dy = xyz[:, 1] - ego_xy[1]
    dist = np.hypot(dx, dy)
    fwd = (dist > 0.2) & (dist <= max_range)
    if fwd.any():
        ang = np.arctan2(dy, dx) - heading
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        fwd &= np.abs(ang) <= np.deg2rad(half_fov_deg)

    if not fwd.any():
        return dict(ego_heading=heading, candidates=[], recommended=None)

    s, d = sem[fwd], dist[fwd]
    cands = []
    for cid in np.unique(s):
        if cid < 0 or cid >= len(CLASS_META):
            continue
        meta = class_meta(int(cid))
        # 软碰撞目标只取「可破坏(frangible)」与「可变形(deformable)」：
        #   - 致命(lethal)/硬刚性(hard)：撞了代价太大，排除
        #   - 可通过(passable)：路面本身是正常行驶面，不是「撞击目标」，排除
        if meta['safety'] not in ('frangible', 'deformable'):
            continue
        m = s == cid
        cands.append(dict(
            class_id=int(cid), class_name=LIDARSEG_32[int(cid)],
            safety=meta['safety'], lethality=meta['lethality'],
            count=int(m.sum()), mean_dist=float(d[m].mean()),
            mean_xy=[float(dx[fwd][m].mean()), float(dy[fwd][m].mean())],
        ))

    cands.sort(key=lambda c: (c['lethality'], c['mean_dist']))
    rec = cands[0] if cands else None
    return dict(ego_heading=heading, candidates=cands[:topk], recommended=rec)


# ---------------- 导出 ----------------
def export_library() -> Dict:
    """导出完整属性库为可 JSON 序列化的 dict（物理 + 视觉 + 类别映射）。"""
    phys = {}
    for name, mid in PHYSICAL_IDS.items():
        m = PHYSICAL_MATERIALS[name]
        phys[mid] = dict(name=name, density=m['density'], friction=m['friction'],
                         restitution=m['restitution'], rigidity=m['rigidity'],
                         destructibility=m['destructibility'], desc=m['desc'])

    vis = {}
    for name, v in VISUAL_MATERIALS.items():
        vis[name] = dict(base_color=list(v['base_color']), metallic=v['metallic'],
                         roughness=v['roughness'])

    class_map = {LIDARSEG_32[i]: dict(CLASS_META[i]) for i in range(len(LIDARSEG_32))}

    return dict(
        collision_safety={s: COLLISION_SAFETY_CN[s] for s in COLLISION_SAFETY},
        physical_materials=phys,
        visual_materials=vis,
        class_meta=class_map,
    )
