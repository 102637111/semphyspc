"""
跨帧实例跟踪：让同一物体在多帧间保持稳定全局 ID，并融合静态物体点云成地图级物体。

匹配：同类 3D 包围盒 IoU + 匈牙利算法（scipy.optimize.linear_sum_assignment）。
  - 静态物体（建筑/树/路沿）：融合点云，中心/包围盒取累积点云；用于地图重建。
  - 动态物体（车/行人/锥桶）：只记录轨迹与最新包围盒，不融合点云。
  - 连续 max_miss 帧未匹配 → 注销（消失/驶出视野）。
"""

from typing import Dict, List
import numpy as np


def bbox_iou(a_min, a_max, b_min, b_max):
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = float(np.prod(np.clip(inter_max - inter_min, 0, None)))
    va = float(np.prod(a_max - a_min))
    vb = float(np.prod(b_max - b_min))
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


class InstanceTracker:
    def __init__(self, iou_thresh: float = 0.2, max_miss: int = 5):
        self.iou_thresh = iou_thresh
        self.max_miss = max_miss
        self.tracks: Dict[int, Dict] = {}
        self.next_tid = 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        """输入一帧检测，返回本帧的 (track_id -> detection) 匹配结果列表。

        detections: list of dict(class_id, class_name, center, bbox_min, bbox_max,
                                 points, is_static)
        """
        tids = list(self.tracks.keys())
        matched = []            # (tid, det_idx)
        unmatched_dets = list(range(len(detections)))

        if tids and detections:
            n_t, n_d = len(tids), len(detections)
            cost = np.full((n_t, n_d), 2.0)   # 2.0 > 任意真实代价（1 - IoU ∈ [0,1]）
            for i, tid in enumerate(tids):
                t = self.tracks[tid]
                for j, d in enumerate(detections):
                    if t['class_id'] != d['class_id']:
                        continue
                    iou = bbox_iou(t['bbox_min'], t['bbox_max'], d['bbox_min'], d['bbox_max'])
                    cost[i, j] = 1.0 - iou
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= 1.0 - self.iou_thresh:
                    matched.append((tids[i], j))
                    if j in unmatched_dets:
                        unmatched_dets.remove(j)

        results = []
        # 更新已匹配
        for tid, j in matched:
            d = detections[j]
            self._absorb(tid, d)
            results.append((tid, d))

        # 新建未匹配的检测
        for j in unmatched_dets:
            tid = self._create(detections[j])
            results.append((tid, detections[j]))

        # 未匹配的 track 计数，超期注销
        matched_tids = {tid for tid, _ in matched}
        for tid in tids:
            if tid not in matched_tids:
                self.tracks[tid]['miss'] += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['miss'] > self.max_miss:
                self.tracks[tid]['state'] = 'lost'
                del self.tracks[tid]

        return results

    def _create(self, d: Dict) -> int:
        tid = self.next_tid
        self.next_tid += 1
        self.tracks[tid] = dict(
            class_id=d['class_id'], class_name=d['class_name'],
            is_static=d['is_static'], state='active', miss=0,
            center=d['center'], bbox_min=d['bbox_min'], bbox_max=d['bbox_max'],
            n_frames=1, n_points=int(d['n_points']),
            fused_points=d['points'] if d['is_static'] else None,
            trajectory=[d['center'].copy()] if not d['is_static'] else [],
        )
        return tid

    def _absorb(self, tid: int, d: Dict):
        t = self.tracks[tid]
        t['miss'] = 0
        t['n_frames'] += 1
        if t['is_static']:
            # 融合静态点云（下采样限幅，避免无限增长）
            fused = np.concatenate([t['fused_points'], d['points']], axis=0)
            if len(fused) > 20000:
                idx = np.random.choice(len(fused), 20000, replace=False)
                fused = fused[idx]
            t['fused_points'] = fused
            t['center'] = fused.mean(axis=0)
            t['bbox_min'] = fused.min(axis=0)
            t['bbox_max'] = fused.max(axis=0)
            t['n_points'] = len(fused)
        else:
            t['center'] = d['center']
            t['bbox_min'] = d['bbox_min']
            t['bbox_max'] = d['bbox_max']
            t['n_points'] = d['n_points']
            t['trajectory'].append(d['center'].copy())

    def summarize(self) -> List[Dict]:
        out = []
        for tid, t in self.tracks.items():
            out.append(dict(
                track_id=tid, class_id=t['class_id'], class_name=t['class_name'],
                is_static=t['is_static'], n_frames=t['n_frames'],
                n_points=t['n_points'], center=t['center'].tolist(),
                bbox_min=t['bbox_min'].tolist(), bbox_max=t['bbox_max'].tolist(),
                trajectory_len=len(t['trajectory']),
            ))
        return out
