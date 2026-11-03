#!/usr/bin/env python3
"""预处理 nuScenes 元数据 → 盲区实验所需紧凑缓存（一次性，约 1 分钟）。

输出 experiments/ray_blindspot/blindspot_meta.pkl:
  keyframes   : [{scene, timestamp, basename}]  按 (scene, timestamp) 排序
  ego_trans   : (N,3) float32   与 keyframes 对齐
  ego_rot     : (N,4) float32   (w,x,y,z) 与 keyframes 对齐
  lidar_extrinsic : (t_cs(3,), q_cs(4,))   LIDAR_TOP 恒定外参（已验 850 token 全同）
"""
import os, json, pickle
import numpy as np

DATA_ROOT = '/mnt/data/nuscenes'
META = os.path.join(DATA_ROOT, 'v1.0-trainval')
OUT = 'experiments/ray_blindspot/blindspot_meta.pkl'


def main():
    print('[1/3] 读 sample_data.json 过滤 LIDAR_TOP 关键帧 ...')
    sd = json.load(open(os.path.join(META, 'sample_data.json')))
    lids = [r for r in sd
            if 'LIDAR_TOP' in r['filename'] and r['fileformat'] == 'pcd' and r['is_key_frame']]
    lids.sort(key=lambda r: (os.path.basename(r['filename']).split('__')[0], r['timestamp']))
    n = len(lids)
    print(f'      LIDAR_TOP 关键帧 {n}')
    calib_tokens = {r['calibrated_sensor_token'] for r in lids}
    del sd

    print('[2/3] 读 ego_pose.json 建 token→pose ...')
    ego = json.load(open(os.path.join(META, 'ego_pose.json')))
    ego_map = {r['token']: (r['translation'], r['rotation']) for r in ego}
    del ego

    print('[3/3] 读外参 + 组装缓存 ...')
    calib = {r['token']: (r['translation'], r['rotation'])
             for r in json.load(open(os.path.join(META, 'calibrated_sensor.json')))}
    t_cs, q_cs = calib[next(iter(calib_tokens))]
    print(f'      LIDAR_TOP 外参 t={t_cs}  q(wxyz)={q_cs}')

    keyframes = []
    ego_trans = np.zeros((n, 3), np.float32)
    ego_rot = np.zeros((n, 4), np.float32)
    for i, r in enumerate(lids):
        keyframes.append({'scene': os.path.basename(r['filename']).split('__')[0],
                          'timestamp': r['timestamp'],
                          'basename': os.path.basename(r['filename'])})
        t, q = ego_map[r['ego_pose_token']]
        ego_trans[i] = t
        ego_rot[i] = q

    meta = {'keyframes': keyframes, 'ego_trans': ego_trans, 'ego_rot': ego_rot,
            'lidar_extrinsic': (np.array(t_cs, np.float32), np.array(q_cs, np.float32))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'wb') as f:
        pickle.dump(meta, f)
    print(f'[完成] 缓存 → {OUT}  ({n} 关键帧)')


if __name__ == '__main__':
    main()
