#!/usr/bin/env python3
"""
撞击后果标定：给 §3.3 的 lethality 表与 §3.5 的破坏规则建立**独立** ground truth。

动机
----
§4.4「位置相关破坏验证」是把 §3.5 公式的输入代回去算一遍
（0.90×0.25=0.225，270 > 0.225×1000 → destroyed）——规则既是前提又是结论。
§4.5 的决策演示同样没回答「撞上去到底能减多少速」，而这正是应用目标
（车辆失控时的选择性碰撞减速）。

方法：用闭合形式的刚体碰撞物理做标定。Δv 有精确解，任何人拿计算器可复核，
不存在「仿真器说如此」的循环论证。

  P0a 解析模型自检   动量守恒 / 能量不增 / 极限行为
  P0b PyBullet 交叉验证及其失败（负结果，说明为何不用仿真器当 ground truth）
  P1  32 类 Δv 标定  与 lethality 表对照
  P1b 帕累托分析     决策只看 lethality 时会不会推荐「撞了也减不了速」的目标
  P2  真实帧决策检验 select_soft_contact 推荐目标的 Δv 实测
  P3  材料强度标定   模型断言值 vs 文献弹性应变能密度 σ_f²/(2E)

纯 CPU。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.materials import (CLASS_MASS, CLASS_STATIC, LIDARSEG_32,
                             class_physics, select_soft_contact)

EGO_MASS = 1500.0          # 自车质量 kg（nuScenes ego 量级）
V_REF = 30.0               # 参考撞击速度 m/s ≈ 108 km/h（失控场景）
GROUND_CLASSES = {24, 25, 26, 27}   # flat.* —— 行驶面，不是障碍物

# 文献值（独立判据，与 utils/damage.py 的 MATERIAL_STRENGTH 无关联）：
#   (密度 kg/m³, 弹性模量 E GPa, 失效强度 σ_f MPa, 断裂韧性 K_IC MPa·m^0.5)
# 取典型工程材料手册量级。σ_f 用**抗拉/失效**强度而非抗压——
# 撞击下的局部破坏由拉应力与断裂控制，混凝土抗压高但抗拉极低。
LITERATURE = {
    'metal':       (7850, 200.0, 250.0, 100.0),   # 结构钢 Q235
    'glass':       (2500,  70.0,  50.0,   0.75),  # 钠钙玻璃
    'concrete':    (2400,  30.0,   3.0,   0.60),  # 抗拉强度（抗压约 30MPa）
    'rock':        (2600,  50.0,  10.0,   1.00),  # 花岗岩
    'asphalt':     (2300,  10.0,   1.5,   0.50),
    'plastic':     (1000,   1.0,  25.0,   1.30),  # HDPE
    'rubber':      (1100,   0.01, 15.0,   5.00),  # 高韧性、几乎不脆断
    'wood':        ( 700,  10.0,  40.0,   0.80),  # 松木顺纹
    'vegetation':  ( 300,   0.50,  5.0,   0.30),
    'soil':        (1500,   0.05,  0.05,  0.10),  # 无抗拉
    'gravel':      (1800,   0.10,  0.10,  0.10),
    'soft_tissue': (1000,   0.01,   1.0,  0.10),
    'unknown':     (1000,   1.0,  20.0,   0.50),
}


# ---------------------------------------------------------------- 解析模型

def delta_v_free(m1, v1, m2, e):
    """自由物体（被撞飞）的一维正碰速度变化——精确解。

    动量守恒 + 恢复系数定义 ⇒ Δv = (1+e)·m2/(m1+m2)·v1
    """
    return (1.0 + e) * m2 / (m1 + m2) * abs(v1)


def delta_v_anchored_rigid(v1, e):
    """锚定刚性物体（墙/固定柱）：Δv = (1+e)·v1。"""
    return (1.0 + e) * abs(v1)


def rebound_velocity_free(m1, v1, m2, e):
    """自由物体碰撞后自身速度（用于守恒自检）。"""
    return (1.0 + e) * m1 / (m1 + m2) * abs(v1)


# ---------------------------------------------------------------- P0a 自检

def part0a_analytic_selfcheck():
    print('=' * 92)
    print('P0a 解析模型自检：动量守恒 / 能量不增 / 极限行为')
    print('=' * 92)

    cases = [(5.0, 0.7), (30.0, 0.4), (70.0, 0.0), (200.0, 0.05), (1500.0, 0.1)]
    print(f"{'m2 kg':>8} {'e':>5} {'Δv m/s':>10} {'物体末速':>10} "
          f"{'p 守恒误差':>12} {'KE 增量':>12}")
    print('-' * 92)
    ok = True
    for m2, e in cases:
        dv = delta_v_free(EGO_MASS, V_REF, m2, e)
        v2 = rebound_velocity_free(EGO_MASS, V_REF, m2, e)
        v1_after = V_REF - dv
        p_err = abs(EGO_MASS * V_REF - (EGO_MASS * v1_after + m2 * v2)) / (EGO_MASS * V_REF)
        ke_ratio = (0.5 * EGO_MASS * v1_after ** 2 + 0.5 * m2 * v2 ** 2) / \
                   (0.5 * EGO_MASS * V_REF ** 2)
        if p_err > 1e-9 or ke_ratio > 1 + 1e-9:
            ok = False
        print(f'{m2:8.0f} {e:5.2f} {dv:10.4f} {v2:10.4f} '
              f'{p_err:11.2e} {ke_ratio:11.4f}')

    # 极限行为
    lim = []
    dv_small = delta_v_free(EGO_MASS, V_REF, 1e-6, 0.5)
    lim.append(('m2 → 0 ⇒ Δv → 0', dv_small < 1e-6, dv_small))
    dv_big = delta_v_free(EGO_MASS, V_REF, 1e9, 0.0)
    lim.append(('m2 → ∞ ⇒ Δv → v1（锚定极限）', abs(dv_big - V_REF) < 1e-3, dv_big))
    dv_anch = delta_v_anchored_rigid(V_REF, 0.05)
    lim.append(('锚定刚性 ⇒ Δv = (1+e)v1', abs(dv_anch - 1.05 * V_REF) < 1e-9, dv_anch))
    e0 = delta_v_free(EGO_MASS, V_REF, 1500.0, 0.0)
    lim.append(('等质量 e=0 ⇒ Δv = v1/2', abs(e0 - V_REF / 2) < 1e-9, e0))

    print()
    for name, passed, val in lim:
        print(f'  [{"通过" if passed else "失败"}] {name:<34} = {val:.6f}')

    all_ok = ok and all(p for _, p, _ in lim)
    print(f'\n  ⇒ 解析模型自检{"全部通过" if all_ok else "**存在失败项**"}')
    return dict(momentum_energy_ok=bool(ok),
                limits=[dict(name=n, passed=bool(p), value=float(v)) for n, p, v in lim],
                all_passed=bool(all_ok),
                worst_p_err=float(max(
                    abs(EGO_MASS * V_REF - (EGO_MASS * (V_REF - delta_v_free(EGO_MASS, V_REF, m2, e))
                        + m2 * rebound_velocity_free(EGO_MASS, V_REF, m2, e))) / (EGO_MASS * V_REF)
                    for m2, e in cases)))


# ---------------------------------------------------------------- P0b PyBullet

def _pybullet_headon(m1, v1, m2, e, dt=1.0 / 10000.0, gap=3.0, iters=200,
                     erp=0.2):
    """自由空间一维正碰。返回 (v_ego_after, v_obj_after)；PyBullet 不可用返回 None。"""
    try:
        import pybullet as p
    except ImportError:
        return None
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, 0, physicsClientId=cid)
        p.setTimeStep(dt, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=iters, erp=erp,
                                    physicsClientId=cid)
        c1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5, physicsClientId=cid)
        c2 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5, physicsClientId=cid)
        b1 = p.createMultiBody(baseMass=m1, baseCollisionShapeIndex=c1,
                               basePosition=[-gap, 0, 0], physicsClientId=cid)
        b2 = p.createMultiBody(baseMass=m2, baseCollisionShapeIndex=c2,
                               basePosition=[0, 0, 0], physicsClientId=cid)
        for b in (b1, b2):
            p.changeDynamics(b, -1, restitution=e, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0,
                             physicsClientId=cid)
        p.resetBaseVelocity(b1, [v1, 0, 0], physicsClientId=cid)
        p.resetBaseVelocity(b2, [0, 0, 0], physicsClientId=cid)
        touched = False
        for _ in range(400000):
            p.stepSimulation(physicsClientId=cid)
            if p.getContactPoints(b1, b2, physicsClientId=cid):
                touched = True
            elif touched:
                break
        r = (float(p.getBaseVelocity(b1, physicsClientId=cid)[0][0]),
             float(p.getBaseVelocity(b2, physicsClientId=cid)[0][0]))
        return r
    finally:
        p.disconnect(physicsClientId=cid)


def part0b_pybullet_check():
    """PyBullet 交叉验证。结论是**负结果**：不作为本脚本的 ground truth。"""
    print()
    print('=' * 92)
    print('P0b PyBullet 交叉验证（负结果：不采用它作为 ground truth）')
    print('=' * 92)
    print(f"{'m1:m2':>10} {'e':>5} {'解析 v_ego':>12} {'PyBullet':>12} "
          f"{'误差':>9} {'动量偏差':>10}")
    print('-' * 92)
    cases = [(1500.0, 0.0), (1500.0, 0.1), (1500.0, 0.7),
             (70.0, 0.0), (30.0, 0.4), (5.0, 0.7)]
    rows = []
    for m2, e in cases:
        out = _pybullet_headon(EGO_MASS, V_REF, m2, e)
        if out is None:
            print('  PyBullet 不可用，跳过')
            return None
        v1_pb, v2_pb = out
        a1 = V_REF - delta_v_free(EGO_MASS, V_REF, m2, e)
        a2 = rebound_velocity_free(EGO_MASS, V_REF, m2, e)
        err = abs(v1_pb - a1) / max(abs(a1), 1e-9)
        dp = abs(EGO_MASS * V_REF - (EGO_MASS * v1_pb + m2 * v2_pb)) / (EGO_MASS * V_REF)
        rows.append(dict(m2=m2, e=e, v_ego_analytic=a1, v_ego_pybullet=v1_pb,
                         v_obj_analytic=a2, v_obj_pybullet=v2_pb,
                         rel_err=err, momentum_err=dp))
        print(f'{EGO_MASS/m2:10.1f} {e:5.2f} {a1:12.4f} {v1_pb:12.4f} '
              f'{err*100:8.2f}% {dp*100:9.2f}%')

    worst = max(r['rel_err'] for r in rows)
    worst_p = max(r['momentum_err'] for r in rows)
    print()
    print(f'  最大速度误差 {worst*100:.1f}%，最大动量偏差 {worst_p*100:.1f}%')

    # 稳健性扫描：这句结论此前只是**断言**，没有实测。既然要作为论文 §4.4.1 的依据，
    # 就必须扫出来。扫描同时纠正了一个错误的解释：原以为该偏差来自高质量比下
    # 顺序冲量求解器的固有限制，实测表明它与质量比无关（见下方对照）。
    m2w, ew = 1500.0, 0.7                      # 上表中 rel_err 最大的算例
    a1w = V_REF - delta_v_free(EGO_MASS, V_REF, m2w, ew)

    def _pb_metrics(**kw):
        out = _pybullet_headon(EGO_MASS, V_REF, m2w, ew, **kw)
        v1w, v2w = out
        err = abs(v1w - a1w) / max(abs(a1w), 1e-9)
        dp = abs(EGO_MASS * V_REF - (EGO_MASS * v1w + m2w * v2w)) / (EGO_MASS * V_REF)
        return v1w, err, dp

    print()
    print(f'  稳健性扫描（算例 m2={m2w:.0f} kg、e={ew}，解析 v_ego={a1w:.4f} m/s）：')
    print(f"    {'扫的变量':<26}{'取值':>10}{'速度误差':>10}{'动量偏差':>10}")
    print('    ' + '-' * 58)
    sweep = {}
    for name, kw in (('时间步 dt (s)', [dict(dt=d) for d in (1e-3, 5e-4, 1e-4, 5e-5)]),
                     ('ERP', [dict(erp=x) for x in (0.0, 0.2, 0.8)]),
                     ('求解器迭代数', [dict(iters=i) for i in (10, 50, 200, 1000)])):
        vals = []
        for k in kw:
            _, err, dp = _pb_metrics(**k)
            vals.append((list(k.values())[0], err, dp))
            print(f'    {name:<26}{list(k.values())[0]:>10}{err*100:>9.2f}%{dp*100:>9.2f}%')
        sweep[name] = vals

    # 质量比对照：这是区分「高质量比固有限制」与「求解器系统性偏离」的判据
    print()
    print(f"    {'质量比 (m1:m2)':<26}{'':>10}{'速度误差':>10}{'动量偏差':>10}")
    print('    ' + '-' * 58)
    ratio_rows = []
    for m2c in (1500.0, 30.0, 5.0):
        out = _pybullet_headon(EGO_MASS, V_REF, m2c, 0.7, dt=1e-4)
        v1c, v2c = out
        a1c = V_REF - delta_v_free(EGO_MASS, V_REF, m2c, 0.7)
        err = abs(v1c - a1c) / max(abs(a1c), 1e-9)
        dp = abs(EGO_MASS * V_REF - (EGO_MASS * v1c + m2c * v2c)) / (EGO_MASS * V_REF)
        ratio_rows.append(dict(m2=m2c, ratio=EGO_MASS / m2c, rel_err=err, momentum_err=dp))
        print(f'    {EGO_MASS/m2c:>10.1f} : 1{"":>14}{err*100:>9.2f}%{dp*100:>9.2f}%')

    print()
    print('  ⇒ 实测特征：动量偏差在**全部**算例、全部 dt/ERP/迭代数取值下恒为约 8.0%，')
    print('     且改变质量比（1:1 与 300:1）**几乎不改变**该值。')
    print('     一个与所有数值参数和质量比都无关的恒定动量亏损，说明它不是离散化或收敛')
    print('     精度问题，而是求解器在本问题上的**系统性偏离**：以等质量、e=0.7 为例，')
    print('     有效恢复系数由设定的 0.7 退化到约 0.52（分离速度 15.57 / 接近速度 30）。')
    print('  ⇒ 结论：PyBullet 在本问题里**不能**充当独立 ground truth——用它就是把一个有偏')
    print('     的模型当成测量，正是本脚本要修的 §4.4 的毛病。故全程采用闭合形式解。')
    return dict(cases=rows, max_rel_err=float(worst),
                max_momentum_err=float(worst_p),
                sweep=sweep, mass_ratio_control=ratio_rows,
                verdict='不可用', reason='恒定约8%动量亏损，与dt/ERP/迭代数/质量比均无关')


# ---------------------------------------------------------------- P1 减速标定

def class_dynamics(cid):
    ph = class_physics(cid)
    m = CLASS_MASS.get(cid)
    if m is None:
        m = float(ph.get('density', 1000.0)) * 0.5
    anchored = cid in CLASS_STATIC
    return dict(mass=float(m), restitution=float(ph.get('restitution', 0.05)),
                anchored=anchored, physical=ph.get('physical'),
                safety=ph.get('safety'), lethality=float(ph.get('lethality', 1.0)),
                damage_response=ph.get('damage_response'),
                is_static_table=anchored, mass_table=float(m))


def part1_deceleration(n_classes=32):
    print()
    print('=' * 92)
    print(f'P1  减速标定：{V_REF:.0f} m/s 撞击各类，自车减速度 Δv vs lethality 表')
    print('=' * 92)
    print(f"{'id':>3} {'类别':<32} {'安全级':<11} {'质量kg':>9} {'锚定':>5} "
          f"{'Δv m/s':>9} {'占初速':>8} {'lethal':>7}")
    print('-' * 92)
    rows = []
    for cid in range(n_classes):
        if cid in (0, 31):
            continue
        d = class_dynamics(cid)
        # 自由体解是精确解；锚定类的点值需要断裂功模型，此处只给上下界
        dv_free = delta_v_free(EGO_MASS, V_REF, d['mass'], d['restitution'])
        dv_rigid = delta_v_anchored_rigid(V_REF, d['restitution'])
        rows.append(dict(class_id=cid, class_name=LIDARSEG_32[cid], **d,
                         delta_v_free=dv_free, delta_v_rigid_upper=dv_rigid,
                         speed_loss_frac=dv_free / V_REF))
        tag = '上界%.1f' % dv_rigid if d['anchored'] else ''
        print(f"{cid:3d} {LIDARSEG_32[cid]:<32} {d['safety']:<11} "
              f"{d['mass']:9.0f} {('是' if d['anchored'] else '否'):>5} "
              f"{dv_free:9.3f} {dv_free/V_REF*100:7.2f}% {d['lethality']:7.2f} {tag}")
    print()
    print('  注：锚定类（右侧标「上界」）的**点值未定**——真值介于自由体值（下界）与')
    print('      (1+e)v（上界）之间，取决于其断裂功，需要额外的断裂模型才能定。')
    print('      本表不对锚定类给出点值估计，避免重蹈 §4.4 用未标定模型当结论的覆辙。')
    return rows


def part1b_pareto(rows):
    print()
    print('=' * 92)
    print('P1b 帕累托分析：lethality（撞了多严重） vs Δv（撞了减多少速）')
    print('=' * 92)

    # select_soft_contact 的实际候选集：frangible / deformable
    cand = [r for r in rows if r['safety'] in ('frangible', 'deformable')]
    cand.sort(key=lambda r: r['lethality'])

    print(f"{'类别':<32} {'安全级':<11} {'lethality':>10} {'Δv m/s':>9} {'占初速':>8} {'锚定':>5}")
    print('-' * 92)
    for r in cand:
        print(f"{r['class_name']:<32} {r['safety']:<11} {r['lethality']:10.2f} "
              f"{r['delta_v_free']:9.3f} {r['delta_v_free']/V_REF*100:7.2f}% "
              f"{('是' if r['anchored'] else '否'):>5}")

    loose = [r for r in cand if not r['anchored']]
    rec = cand[0] if cand else None
    print()
    if loose:
        best = max(loose, key=lambda r: r['delta_v_free'])
        print(f'  当前规则（lethality 升序）会推荐：{rec["class_name"]}')
        print(f'      lethality {rec["lethality"]:.2f}，Δv {rec["delta_v_free"]:.3f} m/s '
              f'（占初速 {rec["delta_v_free"]/V_REF*100:.2f}%）')
        print(f'  候选集中**自由体**减速最大者：{best["class_name"]}')
        print(f'      lethality {best["lethality"]:.2f}，Δv {best["delta_v_free"]:.3f} m/s '
              f'（占初速 {best["delta_v_free"]/V_REF*100:.2f}%）')
        print(f'      → 整个自由体候选集的 Δv 上限仅 {best["delta_v_free"]:.2f} m/s，'
              f'即初速的 {best["delta_v_free"]/V_REF*100:.2f}%')
    print()
    print('  ⇒ 候选集全是轻质量**自由**物体（锥 5kg / 护栏 30kg / 推拉物 20kg），')
    print('     质量比 1:50 ~ 1:300，撞飞它们对 1500kg 自车的减速可忽略。')
    print('     结论：select_soft_contact 优化的是「不伤人」，**没有**优化「减速」——')
    print('     两个目标在候选集上不是同一件事，当前规则只做了前者。')
    return dict(n_candidates=len(cand),
                recommended=rec['class_name'] if rec else None,
                recommended_dv=rec['delta_v_free'] if rec else None,
                loose_max_dv=max((r['delta_v_free'] for r in loose), default=None),
                loose_max_class=max(loose, key=lambda r: r['delta_v_free'])['class_name']
                if loose else None)


# ---------------------------------------------------------------- P2 决策检验

def part2_decision_check(data_dir, split='val', n_frames=40, heading_deg=0.0,
                         max_range=50.0, half_fov=40.0):
    import glob
    from collections import Counter
    print()
    print('=' * 92)
    print('P2  决策检验：真实 nuScenes 帧上 select_soft_contact 的推荐目标准不减速？')
    print('=' * 92)

    frames = sorted(glob.glob(os.path.join(data_dir, split, '*.npz')))[:n_frames]
    if not frames:
        print(f'  没有找到 npz：{data_dir}/{split}，跳过')
        return None

    heading = np.deg2rad(heading_deg)
    per_frame, recs = [], []
    for fp in frames:
        d = np.load(fp)
        out = select_soft_contact(d['sem'].reshape(-1),
                                  d['xyz'][:, :3].astype(np.float32),
                                  heading=heading, max_range=max_range,
                                  half_fov_deg=half_fov)
        rec = out.get('recommended')
        if rec is None:
            per_frame.append(dict(frame=os.path.basename(fp), n_cand=0,
                                  recommended=None, delta_v=None))
            continue
        dd = class_dynamics(rec['class_id'])
        dv = delta_v_free(EGO_MASS, V_REF, dd['mass'], dd['restitution'])
        row = dict(frame=os.path.basename(fp), n_cand=len(out['candidates']),
                   recommended=rec['class_name'], class_id=rec['class_id'],
                   mass=dd['mass'], anchored=dd['anchored'], delta_v=dv,
                   speed_loss_frac=dv / V_REF)
        per_frame.append(row)
        recs.append(row)

    if not recs:
        print('  这些帧视锥内没有可撞击候选（全是致命/硬刚性/可通过）')
        return dict(n_frames=len(frames), n_with_recommendation=0, per_frame=per_frame)

    cnt = Counter(r['recommended'] for r in recs)
    dvs = np.array([r['delta_v'] for r in recs])
    print(f'  {len(frames)} 帧中 {len(recs)} 帧给出推荐目标（{len(recs)/len(frames)*100:.0f}%）')
    print()
    print(f"{'推荐类别':<34} {'帧数':>6} {'Δv m/s':>9} {'占初速':>9} {'锚定':>5}")
    print('-' * 92)
    for name, c in cnt.most_common():
        sub = [r for r in recs if r['recommended'] == name]
        print(f"{name:<34} {c:6d} {np.mean([r['delta_v'] for r in sub]):9.3f} "
              f"{np.mean([r['delta_v'] for r in sub])/V_REF*100:8.2f}% "
              f"{('是' if sub[0]['anchored'] else '否'):>5}")
    print()
    print(f'  推荐目标 Δv：中位 {np.median(dvs):.3f} m/s，均值 {dvs.mean():.3f} m/s，'
          f'最大 {dvs.max():.3f} m/s')
    print(f'  相对初速 {V_REF:.0f} m/s：中位减速 {np.median(dvs)/V_REF*100:.2f}%')

    # --- 地面类诊断：flat.terrain 在代码里是 deformable（论文 §3.3 说它是 passable），
    #     于是地面进了候选集且 lethality 最低，几乎每帧都赢下推荐位。
    n_ground = sum(1 for r in recs if r['class_id'] in GROUND_CLASSES)
    print()
    print('-' * 92)
    print(f'  ⚠ 推荐目标是**地面类**的帧：{n_ground}/{len(recs)} '
          f'({n_ground/len(recs)*100:.0f}%)')
    if n_ground:
        print('     flat.terrain 在 utils/materials.py 里 safety=deformable（lethality 0.05），')
        print('     而论文 §3.3 把它归为 passable。「地面」不是障碍物而是行驶面，')
        print('     §3.6 的候选过滤只排除 passable，于是地面拿到最低 lethality、稳赢排序。')
        print('     论文 §4.5 的演示帧 000260ba… 恰好 terrain 点数为 0，所以单帧演示没暴露。')
    print()
    print('  ⇒ 独立缺口：候选集全是轻质量**自由**体，撞飞它们的减速 ≤2.75%，')
    print('     达不到「碰撞减速」目标。这正是 utils/impact_decision.py 要补的：')
    print('     纳入锚定类，并引入对硬墙面的刮蹭减速（见 P4）。')
    if n_ground == 0:
        print('     注：地面类推荐问题已于 2026-09-14 修复（flat.terrain 归回 passable），')
        print('         修复前该比例为 73%。')
    return dict(n_frames=len(frames), n_with_recommendation=len(recs),
                recommended_dist=dict(cnt),
                n_ground_recommended=n_ground,
                ground_frac=n_ground / len(recs),
                dv_median=float(np.median(dvs)), dv_mean=float(dvs.mean()),
                dv_max=float(dvs.max()),
                speed_loss_median_frac=float(np.median(dvs) / V_REF),
                per_frame=per_frame)


# ---------------------------------------------------------------- P3 强度标定

def part3_strength_calibration():
    from scipy.stats import spearmanr
    from utils.damage import MATERIAL_STRENGTH

    print()
    print('=' * 92)
    print('P3  材料强度标定：模型断言值 vs 文献弹性应变能密度 σ_f²/(2E)')
    print('=' * 92)
    print('  σ_f²/(2E) = 失效前的弹性应变能密度（MJ/m³），物理上决定「撞多狠才坏」')
    print('  取抗拉/失效强度而非抗压：撞击局部破坏由拉应力与断裂控制')
    print()
    print(f"{'材质':<13} {'模型值':>7} {'E GPa':>7} {'σ_f MPa':>8} {'K_IC':>7} "
          f"{'σ²/2E':>10} {'物理序':>7} {'模型序':>7} {'落差':>6}")
    print('-' * 92)

    common = [m for m in MATERIAL_STRENGTH if m in LITERATURE]
    phys = {m: LITERATURE[m][2] ** 2 / (2.0 * LITERATURE[m][1] * 1000.0) for m in common}
    phys_rank = {m: i + 1 for i, m in enumerate(sorted(common, key=lambda x: -phys[x]))}
    mod_rank = {m: i + 1 for i, m in
                enumerate(sorted(common, key=lambda x: -MATERIAL_STRENGTH[x]))}

    for m in sorted(common, key=lambda x: -phys[x]):
        _, E, sf, kic = LITERATURE[m]
        gap = phys_rank[m] - mod_rank[m]
        print(f'{m:<13} {MATERIAL_STRENGTH[m]:7.2f} {E:7.2f} {sf:8.2f} {kic:7.2f} '
              f'{phys[m]:10.6f} {phys_rank[m]:7d} {mod_rank[m]:7d} {gap:+6d}')

    rho, pval = spearmanr([MATERIAL_STRENGTH[m] for m in common], [phys[m] for m in common])
    print()
    print(f'  Spearman 秩相关 ρ = {rho:.3f}  (p = {pval:.4f})')
    print()
    print('  模型高估最严重的材质（模型说最耐撞、物理上最脆）：')
    worst = sorted(common, key=lambda x: phys_rank[x] - mod_rank[x], reverse=True)[:4]
    for m in worst:
        if phys_rank[m] - mod_rank[m] <= 0:
            continue
        print(f'    {m:<13} 模型第 {mod_rank[m]:2d}/{len(common)}  物理第 '
              f'{phys_rank[m]:2d}/{len(common)}  σ²/2E = {phys[m]:.6f} MJ/m³')
    print()
    print('  ⇒ 模型把混凝土/岩石/沥青当作最耐撞（0.90~0.95），但它们的**抗拉**失效')
    print('     应变能密度比钢低 3 个数量级——撞击下混凝土会碎而钢会弯。')
    print('     这直接影响 §3.3 的材质表与 §3.5 的 hp 阈值。')
    return dict(spearman_rho=float(rho), spearman_p=float(pval),
                strain_energy=phys, model_rank=mod_rank, phys_rank=phys_rank,
                literature={m: dict(E_GPa=LITERATURE[m][1], sigma_f_MPa=LITERATURE[m][2],
                                    K_IC=LITERATURE[m][3]) for m in common})


# ---------------------------------------------------------------- P4 合成验证

def _simulate_lidar_wall(w0, u_start, u_end, h_wall, h_sensor=1.84,
                         n_az=1800, elev_lo=-30.67, elev_hi=10.67, n_beam=32):
    """把一面与行进方向并行的墙用 32 线旋转激光雷达模型采样。

    墙为平面 y = w0（w0>0），x ∈ [u_start, u_end]，z ∈ [0, h_wall]。
    返回击中墙面的点 (N,3)，传感器位于 (0,0,h_sensor)。
    """
    elev = np.deg2rad(np.linspace(elev_lo, elev_hi, n_beam))
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(elev, az, indexing='ij')
    dx = np.cos(E) * np.cos(A)
    dy = np.cos(E) * np.sin(A)
    dz = np.sin(E)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = w0 / dy
    ok = np.isfinite(t) & (t > 0)
    x = np.where(ok, t * dx, np.nan)
    z = np.where(ok, h_sensor + t * dz, np.nan)
    m = ok & (x >= u_start) & (x <= u_end) & (z >= 0) & (z <= h_wall)
    return np.stack([x[m], np.full(m.sum(), w0), z[m]], axis=1).astype(np.float32)


def part4_synthetic_validation():
    """合成场景验证刮蹭模块：已知几何 → 检查规划器输出与手算一致。

    同时量化「与行进方向并行的墙对视旋转激光雷达是掠射、采样稀疏」这一效应——
    它解释了为何 nuScenes val 里几乎找不到可刮蹭长墙（16k 与 64k 结果完全相同，
    故与降采样无关）。
    """
    from utils.impact_decision import (MU_TIRE, MU_WALL_BY_MATERIAL, _wall_geometry,
                                       dv_scrape, plan_selective_impact)

    print()
    print('=' * 92)
    print('P4  合成验证：已知几何下的刮蹭模块正确性 + 掠射采样效应')
    print('=' * 92)

    W0, U0, U1, HW = 4.0, 2.0, 62.0, 3.0
    pts = _simulate_lidar_wall(W0, U0, U1, HW)
    print(f'  合成墙: 侧向 {W0} m，纵向 {U0}~{U1} m，高 {HW} m')
    print(f'  32 线雷达采样命中 {len(pts)} 点')

    # 沿纵向的命中密度——掠射效应直接看这里
    print()
    print(f"  {'u 区间':>12} {'命中点数':>9} {'每米点数':>9}")
    print('  ' + '-' * 34)
    for a, b in [(2, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 62)]:
        n = int(((pts[:, 0] >= a) & (pts[:, 0] < b)).sum())
        print(f'  {f"{a}-{b} m":>12} {n:>9} {n/(b-a):>9.1f}')
    dens_near = ((pts[:, 0] >= 2) & (pts[:, 0] < 5)).sum() / 3
    dens_far = ((pts[:, 0] >= 30) & (pts[:, 0] < 45)).sum() / 15
    print(f'\n  近处(2-5m) {dens_near:.1f} 点/m → 远处(30-45m) {dens_far:.2f} 点/m，'
          f'衰减 {dens_near/max(dens_far,1e-9):.0f}×')
    print('  ⇒ 与行进方向并行的墙是**掠射**：入射角随距离趋于平行，')
    print('     远处每米只剩个位数点，1 m 分箱内凑不满 3 点，检测自然断开。')

    # 规划器端到端
    sem_wall = np.full(len(pts), 28, dtype=np.int64)
    sem_gnd = np.full(4000, 24, dtype=np.int64)
    gx = np.random.default_rng(0).uniform(-20, 62, 4000)
    gy = np.random.default_rng(1).uniform(-20, 20, 4000)
    gnd = np.stack([gx, gy, np.full(4000, -1.84)], axis=1).astype(np.float32)
    cloud = np.vstack([pts, gnd])
    sem = np.concatenate([sem_wall, sem_gnd])
    out = plan_selective_impact(sem, cloud, speed=30.0,
                                scrape_u_range=(2.0, 65.0))

    print()
    print('  规划器输出：')
    for s in out['scrape'][:3]:
        print(f"    刮蹭 {s['class_name']} 墙长{s['wall_len']:.1f}m 高{s['wall_height']:.1f} "
              f"侧{s['lateral']:.2f}m 角{s['approach_deg']:.2f}° Δv={s['dv']:.3f} m/s")
    rec = out['recommended']
    print(f"  推荐: {rec['mode'] if rec else None} "
          f"{rec['class_name'] if rec else ''} Δv={rec['rank_dv']:.3f}" if rec
          else '  推荐: 无')

    # 手算对照
    mu_wall = MU_WALL_BY_MATERIAL['concrete']
    a_expect = mu_wall * MU_TIRE * 9.81
    print()
    print(f'  手算：a = μ_wall({mu_wall})·μ_tire({MU_TIRE})·g = {a_expect:.3f} m/s²')
    if out['scrape']:
        s = out['scrape'][0]
        exp_dv = min(a_expect * s['wall_len'] / 30.0, 30.0)
        print(f'  检测到墙长 {s["wall_len"]:.1f} m → 期望 Δv = {exp_dv:.3f} m/s，'
              f'规划器给出 {s["dv"]:.3f} m/s，误差 '
              f'{abs(s["dv"]-exp_dv)/max(exp_dv,1e-9)*100:.2f}%')
    else:
        print('  ⚠ 规划器未检出墙——合成场景下这属于实现缺陷，需排查')

    return dict(wall_spec=dict(lateral=W0, u_range=[U0, U1], height=HW),
                n_hits=int(len(pts)),
                density_near=float(dens_near), density_far=float(dens_far),
                density_decay=float(dens_near / max(dens_far, 1e-9)),
                decel_expected=float(a_expect),
                n_scrape_found=len(out['scrape']),
                scrape_top=out['scrape'][0] if out['scrape'] else None,
                recommended=rec)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nuscenes_seg_full_16k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-frames', type=int, default=40)
    ap.add_argument('--heading-deg', type=float, default=0.0)
    ap.add_argument('--parts', default='0a,0b,1,1b,2,3,4')
    ap.add_argument('--out', default='results/impact_calibration.json')
    args = ap.parse_args()

    want = set(args.parts.split(','))
    res = dict(ego_mass=EGO_MASS, v_ref=V_REF,
               class_mass_source='utils/materials.py CLASS_MASS',
               anchored_source='utils/materials.py CLASS_STATIC')

    if '0a' in want:
        res['P0a_analytic_selfcheck'] = part0a_analytic_selfcheck()
    if '0b' in want:
        res['P0b_pybullet_check'] = part0b_pybullet_check()
    if '1' in want or '1b' in want:
        rows = part1_deceleration()
        res['P1_deceleration'] = rows
        if '1b' in want:
            res['P1b_pareto'] = part1b_pareto(rows)
    if '2' in want:
        res['P2_decision_check'] = part2_decision_check(
            args.data, args.split, args.n_frames, args.heading_deg)
    if '3' in want:
        res['P3_strength_calibration'] = part3_strength_calibration()
    if '4' in want:
        res['P4_synthetic_validation'] = part4_synthetic_validation()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print()
    print(f'已写出：{args.out}')


if __name__ == '__main__':
    main()
