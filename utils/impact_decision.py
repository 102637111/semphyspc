"""
选择性撞击规划：从「只挑最不致命的」升级为「在安全约束下最大化减速」。

动机（见 scripts/impact_calibration.py 的实测）
----------------------------------------------
原 select_soft_contact 只按 lethality 升序选，候选集限于 frangible/deformable
的**轻质量自由体**（锥 5kg / 护栏 30kg / 推拉物 20kg）。30 m/s 下撞这些的减速
上限只有 0.824 m/s，即初速的 2.75%——满足「不伤人」，但没达成「碰撞减速」。
而应用目标（车辆失控时的选择性碰撞减速）需要的是后者。

本模块引入两类接触模式，并允许候选集纳入**锚定类**：

  模式 A  正碰软目标 (head-on)
    自由物体：Δv = (1+e)·m2/(m1+m2)·v           —— 精确解
    锚定物体：Δv ∈ [自由体值, (1+e)v]           —— 点值需断裂功模型，本模块只给区间
              （不给点值：拒绝用未标定模型当结论，那是 §4.4 的毛病）

  模式 B  刮蹭硬面 (glancing scrape)
    对**硬刚性锚定面**（墙体/建筑/固定护栏）以小角度侧向接触，靠持续摩擦减速：
        a = μ_wall · μ_tire · g
        Δv = a · L / v      （L 为可用刮蹭行程，上限为 v）
    法向力上限来自轮胎侧向力上限 μ_tire·m·g，故 a 与质量无关。
    轮胎侧向摩擦 μ_tire≈0.9、车体-混凝土 μ_wall≈0.50 ⇒ a ≈ 4.4 m/s²，
    50 m 行程在 30 m/s 下可减 7.35 m/s（24.5%），是软目标正碰上限的约 9 倍。
    这是「撞墙」与「蹭墙」的区别：正碰致命，刮蹭可控。

决策：在 risk ≤ risk_limit 的约束下最大化 Δv。两类模式的风险量纲不同——
正碰用类别的 lethality；刮蹭用接近角（越平缓越安全），不用 lethality，
因为轻蹭墙面并不是「撞墙」。

已知不一致（本模块不擅自改，仅记录）
------------------------------------
utils/materials.py 同时有 CLASS_MASS[30]=200（植被）与 30 ∈ CLASS_STATIC，
DamageModel 会同时报告 mass=200 与 is_static=True，物理上矛盾。
本模块按 CLASS_STATIC 处理（树是扎根的，锚定才符合物理）。
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .materials import (CLASS_MASS, CLASS_META, CLASS_STATIC, LIDARSEG_32,
                        class_meta, class_physics)

G = 9.81

# --- 刮蹭模型参数（可标定） ---
MU_TIRE = 0.90          # 干沥青轮胎侧向摩擦系数上限
MU_WALL_DEFAULT = 0.50  # 车身与墙面摩擦系数兜底
MU_WALL_BY_MATERIAL = {
    'concrete':   0.50,   # 混凝土墙
    'metal':      0.30,   # 钢制护栏（钢-钢偏低）
    'plastic':    0.35,
    'wood':       0.45,
    'rock':       0.55,
    'asphalt':    0.55,
    'vegetation': 0.60,
    'glass':      0.25,
}
GROUND_CLASSES = {24, 25, 26, 27}   # flat.* 行驶面，不是接触目标

# 锚定类的质量先验（CLASS_MASS 里没有的锚定类用密度×兜底体积）
_FALLBACK_VOLUME = 0.5


# ------------------------------------------------------------------ 物理

def dv_headon_free(m1: float, v: float, m2: float, e: float) -> float:
    """自由物体正碰：Δv = (1+e)·m2/(m1+m2)·v（精确解）。"""
    return (1.0 + e) * m2 / (m1 + m2) * abs(v)


def dv_headon_anchored_upper(v: float, e: float) -> float:
    """锚定物体正碰的上界（刚性锚定）：Δv = (1+e)·v。"""
    return (1.0 + e) * abs(v)


def scrape_deceleration(physical: str, mu_tire: float = MU_TIRE) -> float:
    """刮蹭的持续减速度 a = μ_wall·μ_tire·g（m/s²，与质量无关）。"""
    mu_wall = MU_WALL_BY_MATERIAL.get(physical, MU_WALL_DEFAULT)
    return mu_wall * mu_tire * G


def dv_scrape(physical: str, v: float, length: float,
              mu_tire: float = MU_TIRE) -> float:
    """刮蹭行程 length 米后的总减速量，上限为 v（撞停即止）。"""
    if v <= 0 or length <= 0:
        return 0.0
    a = scrape_deceleration(physical, mu_tire)
    return float(min(a * length / v, v))


def scrape_risk(approach_deg: float, max_approach_deg: float = 20.0) -> float:
    """刮蹭风险：接近角越陡越接近正碰，风险越高（0~1）。

    用 (α/α_max)² 而非线性：小角度确实安全，超过阈值迅速恶化。
    """
    if max_approach_deg <= 0:
        return 1.0
    r = abs(approach_deg) / max_approach_deg
    return float(min(r * r, 1.0))


# ------------------------------------------------------------------ 几何

def _wall_geometry(pts: np.ndarray, heading: float, bin_w: float = 1.0,
                   tol_step: float = 0.35, min_height: float = 0.6):
    """判断一组点是否构成可刮蹭的、与行进方向大致平行的竖直墙面。

    不对整类点做单次 PCA——同一类在走廊内的点会跨越朝向不同的多面墙并混入
    屋顶杂物，PCA 主轴无意义（实测 39/40 帧被误判为「非竖直面」）。

    改用墙面本身的定义：转到航向坐标系 (u 前向, w 侧向)，按 u 分箱；
    先按点数选占优的**一侧**（走廊两侧可能都有墙，取中位数会让两侧互相抵消），
    再对该侧各箱的 med_w(u) 拟合直线——**斜率即墙相对行进方向的夹角**，
    残差小说明点确实落在一条线上（是墙面而非散点）。

    Returns: dict(ok, reason, length, lateral, height, approach_deg, n_pts)
    """
    if len(pts) < 30:
        return dict(ok=False, reason='点太少')

    c, s = math.cos(heading), math.sin(heading)
    u = pts[:, 0] * c + pts[:, 1] * s            # 前向
    w = -pts[:, 0] * s + pts[:, 1] * c           # 侧向（左正）
    z = pts[:, 2]

    u0, u1 = float(u.min()), float(u.max())
    if u1 - u0 < bin_w:
        return dict(ok=False, reason='纵向跨度不足')
    nb = max(int((u1 - u0) / bin_w), 1)
    idx = np.clip(((u - u0) / bin_w).astype(int), 0, nb - 1)

    med_w = np.full(nb, np.nan)
    spread = np.full(nb, np.inf)
    height = np.zeros(nb)
    cnt = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b] >= 3:
            med_w[b] = float(np.median(w[m]))
            spread[b] = float(np.percentile(w[m], 90) - np.percentile(w[m], 10))
            height[b] = float(z[m].max() - z[m].min())

    good = cnt >= 3
    if good.sum() < 3:
        return dict(ok=False, reason='有效纵向分箱不足')

    # 按点数选占优的一侧——两侧都有墙时不能取全局中位数（会互相抵消到 0）
    pos_mass = float(cnt[good & (med_w > 0)].sum())
    neg_mass = float(cnt[good & (med_w < 0)].sum())
    sel = good & ((med_w > 0) if pos_mass >= neg_mass else (med_w < 0))
    if sel.sum() < 3:
        return dict(ok=False, reason='单侧有效分箱不足')

    us = u0 + (np.arange(nb) + 0.5) * bin_w
    on_base = sel & (spread < 1.5) & (height >= min_height)
    if on_base.sum() < 3:
        return dict(ok=False, reason='无合格分箱')

    # 局部连续性：相邻箱横向漂移小于 tol_step 才连成一片。
    # 用局部而非全局直线拟合——真实建筑 60m 内有转角/凹进，全局拟合必然失败；
    # 而转角处**本就该断开**（转角蹭不过去），局部判据正是物理上正确的那个。
    link = np.zeros(max(nb - 1, 0), dtype=bool)
    for b in range(1, nb):
        if on_base[b] and on_base[b - 1] and abs(med_w[b] - med_w[b - 1]) < tol_step:
            link[b - 1] = True

    best_len = cur = 0
    best_start = cur_start = 0
    for b in range(len(link)):
        if link[b]:
            if cur == 0:
                cur_start = b
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, cur_start
        else:
            cur = 0
    n_bins = best_len + 1                      # 连续 link 覆盖 best_len+1 个箱
    if n_bins < 3:
        return dict(ok=False, reason='连续墙面过短')

    seg = slice(best_start, best_start + n_bins)
    u_seg, w_seg = us[seg], med_w[seg]
    # 近似的接近角：链内横向漂移率（局部直线，不再要求全局共线）
    slope = float(np.polyfit(u_seg, w_seg, 1)[0]) if n_bins >= 2 else 0.0
    return dict(ok=True,
                length=float(n_bins * bin_w),
                lateral=float(w_seg[0]),
                height=float(np.median(height[seg])),
                approach_deg=float(math.degrees(math.atan(slope))),
                n_pts=int(cnt[seg].sum()))


# ------------------------------------------------------------------ 规划

def _cone_mask(xyz, ego_xy, heading, max_range, half_fov_deg):
    dx = xyz[:, 0] - ego_xy[0]
    dy = xyz[:, 1] - ego_xy[1]
    dist = np.hypot(dx, dy)
    m = (dist > 0.2) & (dist <= max_range)
    if m.any():
        ang = np.arctan2(dy, dx) - heading
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        m &= np.abs(ang) <= np.deg2rad(half_fov_deg)
    return m, dist


def _corridor_mask(xyz, ego_xy, heading, u_range, w_range):
    """侧向走廊：|w| ∈ w_range 且 u ∈ u_range（航向坐标系）。

    刮蹭要用**另一套搜索区域**，不能复用前向视锥：正碰看正前方，
    而可刮蹭的墙必须与路径**并行**、位于车侧——前向锥看不到它。
    """
    dx = xyz[:, 0] - ego_xy[0]
    dy = xyz[:, 1] - ego_xy[1]
    c, s = math.cos(heading), math.sin(heading)
    u = dx * c + dy * s
    w = -dx * s + dy * c
    m = (u >= u_range[0]) & (u <= u_range[1]) & \
        (np.abs(w) >= w_range[0]) & (np.abs(w) <= w_range[1])
    return m


def plan_selective_impact(sem_labels, xyz, ego_xy=(0.0, 0.0), heading=0.0,
                          speed=30.0, ego_mass=1500.0,
                          max_range=50.0, half_fov_deg=40.0,
                          risk_limit=0.60, max_approach_deg=20.0,
                          min_wall_len=10.0, topk=5,
                          scrape_u_range=(3.0, 60.0), scrape_w_range=(1.5, 8.0)):
    """在安全约束下最大化减速的选择性撞击规划。

    Args:
        speed:           自车当前速度 m/s（Δv 随速度变化，必须传入）
        risk_limit:      可接受风险上限（正碰用 lethality，刮蹭用接近角风险）
        max_approach_deg:刮蹭允许的最大接近角
        min_wall_len:    可刮蹭墙面的最小行程 m
        scrape_u_range:  刮蹭走廊的前向范围 (m)
        scrape_w_range:  刮蹭走廊的侧向距离范围 (m)——伸手可及、值得蹭的墙

    Returns:
        dict(headon=[...], scrape=[...], options=[...], recommended=..., note=...)
    """
    sem = np.asarray(sem_labels).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    v = float(speed)

    cone, dist = _cone_mask(xyz, ego_xy, heading, max_range, half_fov_deg)
    headon, scrape = [], []

    if cone.any():
        s_c, d_c, p_c = sem[cone], dist[cone], xyz[cone]

        # ---- 模式 A：正碰软目标（frangible / deformable） ----
        for cid in np.unique(s_c):
            cid = int(cid)
            if cid in GROUND_CLASSES or cid < 0 or cid >= len(CLASS_META):
                continue
            meta = class_meta(cid)
            if meta['safety'] not in ('frangible', 'deformable'):
                continue
            leth = float(meta['lethality'])
            if leth > risk_limit:
                continue
            ph = class_physics(cid)
            m2 = CLASS_MASS.get(cid)
            if m2 is None:
                m2 = float(ph.get('density', 1000.0)) * _FALLBACK_VOLUME
            e = float(ph.get('restitution', 0.05))
            anchored = cid in CLASS_STATIC
            dv_lo = dv_headon_free(ego_mass, v, float(m2), e)
            dv_hi = dv_headon_anchored_upper(v, e) if anchored else dv_lo
            m = s_c == cid
            headon.append(dict(mode='headon', class_id=cid,
                               class_name=LIDARSEG_32[cid],
                               safety=meta['safety'], lethality=leth,
                               anchored=bool(anchored), mass=float(m2),
                               dv_lo=dv_lo, dv_hi=dv_hi,
                               rank_dv=dv_lo,      # 按下界排序（保守）
                               risk=leth,
                               count=int(m.sum()),
                               mean_dist=float(d_c[m].mean()),
                               mean_xy=[float(p_c[m][:, 0].mean()),
                                        float(p_c[m][:, 1].mean())]))

    # ---- 模式 B：刮蹭硬面（hard 且锚定），用**侧向走廊**而非前向锥 ----
    corridor = _corridor_mask(xyz, ego_xy, heading, scrape_u_range, scrape_w_range)
    if corridor.any():
        s_k, p_k = sem[corridor], xyz[corridor]
        for cid in np.unique(s_k):
            cid = int(cid)
            if cid in GROUND_CLASSES or cid < 0 or cid >= len(CLASS_META):
                continue
            meta = class_meta(cid)
            # 刮蹭需要**硬**面提供持续摩擦：可变形面（灌木）会屈服，
            # 只给一次冲量而非持续阻力，不适用库仑摩擦模型。
            if meta['safety'] != 'hard':
                continue
            if cid not in CLASS_STATIC:
                continue          # 刮蹭要求面是锚定的
            m = s_k == cid
            g = _wall_geometry(p_k[m], heading)
            if not g['ok']:
                continue
            if g['approach_deg'] > max_approach_deg:
                continue
            if g['length'] < min_wall_len:
                continue
            ph = class_physics(cid)
            dv = dv_scrape(ph.get('physical', 'unknown'), v, g['length'])
            risk = scrape_risk(g['approach_deg'], max_approach_deg)
            if risk > risk_limit:
                continue
            scrape.append(dict(mode='scrape', class_id=cid,
                               class_name=LIDARSEG_32[cid],
                               safety='hard',       # 刮蹭目标限定 hard 类，供可视化着色
                               physical=ph.get('physical'),
                               approach_deg=g['approach_deg'],
                               wall_len=g['length'],
                               wall_height=g['height'],
                               lateral=g['lateral'],
                               n_pts=g['n_pts'],
                               decel=scrape_deceleration(ph.get('physical', 'unknown')),
                               dv=dv, rank_dv=dv, risk=risk,
                               count=int(m.sum()),
                               mean_dist=float(np.abs(g['lateral'])),
                               mean_xy=[float(p_k[m][:, 0].mean()),
                                        float(p_k[m][:, 1].mean())]))

    options = headon + scrape
    options.sort(key=lambda o: (-o['rank_dv'], o['risk']))
    rec = options[0] if options else None

    note = ('无可选方案：视锥内既无低致命软目标，也无满足接近角/行程的锚定墙面'
            if not options else
            f'{len(headon)} 个正碰候选 + {len(scrape)} 个刮蹭候选，'
            f'最优 Δv={rec["rank_dv"]:.2f} m/s（{rec["rank_dv"]/v*100:.1f}% 初速）')

    return dict(speed=v, risk_limit=risk_limit,
                max_approach_deg=max_approach_deg,
                headon=sorted(headon, key=lambda o: -o['rank_dv'])[:topk],
                scrape=sorted(scrape, key=lambda o: -o['dv'])[:topk],
                options=options[:topk], recommended=rec, note=note)
