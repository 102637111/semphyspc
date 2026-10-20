#!/usr/bin/env python3
"""训练监控 + 自动关机看门狗（nohup 后台运行，可跨会话存活）。

监控 /tmp/seg_full.log：
  - 训练自然结束（出现 "训练结束"）              → 关机
  - mIoU 平台期（连续 PLATEAU_EPOCHS 个 epoch 未刷新最佳 mIoU，且已过 MIN_EPOCH）→ 关机
  - 检测到异常（Traceback/RuntimeError/CUDA error）→ 不关机，仅退出

关机：echo <PASSWORD> | sudo -S shutdown -h now（密码为会话内用户提供，仅用于本次关机）。
"""
import re, sys, time, os, subprocess

LOG = '/tmp/seg_full.log'
PLATEAU_EPOCHS = 30      # 连续多少 epoch 无提升判定为平台期
MIN_EPOCH = 60           # 至少训到该 epoch 才考虑平台期（避免过早关机）
CHECK_INTERVAL = 20      # 轮询间隔（秒）
GRACE_SECONDS = 90       # 触发关机前留的时间，让当前 epoch 存完 ckpt

PASSWORD = '102637'

best_miou = -1.0
best_epoch = -1
cur_epoch = 0
last_size = 0


def log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def do_shutdown(reason):
    log(f'触发关机：{reason}')
    log(f'等待 {GRACE_SECONDS}s 让当前 epoch 保存 ckpt ...')
    time.sleep(GRACE_SECONDS)
    subprocess.run(f'echo {PASSWORD} | sudo -S shutdown -h now', shell=True)
    sys.exit(0)


def main():
    global best_miou, best_epoch, cur_epoch, last_size
    log(f'看门狗启动，监控 {LOG}')
    log(f'平台期判定：>={MIN_EPOCH} epoch 且连续 {PLATEAU_EPOCHS} epoch 无最佳提升 → 关机')

    if os.path.exists(LOG):
        last_size = os.path.getsize(LOG)  # 只监控启动后的新日志

    while True:
        time.sleep(CHECK_INTERVAL)
        try:
            with open(LOG, 'r') as f:
                f.seek(last_size)
                new = f.read()
                last_size = f.tell()
        except FileNotFoundError:
            continue

        if not new:
            continue

        # 异常 → 不关机，仅退出
        if any(k in new for k in ('Traceback', 'RuntimeError', 'CUDA error', 'CUDA out of memory')):
            log('检测到训练异常，不关机，看门狗退出')
            sys.exit(0)

        for line in new.splitlines():
            m = re.search(r'\[Validation\] EPOCH:\s*(\d+)\s+mIoU = ([\d.]+)', line)
            if m:
                ep = int(m.group(1))
                miou = float(m.group(2))
                cur_epoch = max(cur_epoch, ep)
                if miou > best_miou + 0.0005:
                    best_miou = miou
                    best_epoch = ep
                    log(f'新最佳 mIoU = {miou:.4f} @ epoch {ep}')

            if '训练结束' in line:
                do_shutdown('训练完成（200 epoch）')
                sys.exit(0)

        # 平台期判定
        if best_epoch > 0 and cur_epoch >= MIN_EPOCH and (cur_epoch - best_epoch) >= PLATEAU_EPOCHS:
            do_shutdown(
                f'mIoU 平台期：最佳 {best_miou:.4f} @ epoch {best_epoch}，'
                f'已连续 {cur_epoch - best_epoch} epoch 无提升（当前 epoch {cur_epoch}）'
            )
            sys.exit(0)


if __name__ == '__main__':
    main()
