#!/usr/bin/env python3
"""PointMambaSeg 逐点语义分割训练/测试入口（单卡）。

用法:
    python main_seg.py --config cfgs/seg_nuscenes.yaml --exp_name seg_v1
    python main_seg.py --config cfgs/seg_nuscenes.yaml --test --ckpts <path>/ckpt-best.pth
    python main_seg.py --config cfgs/seg_nuscenes.yaml --resume --exp_name seg_v1
"""

import os, time, datetime, random, argparse, glob
from pathlib import Path
import numpy as np
import torch
from tensorboardX import SummaryWriter

from utils.config import get_config, log_args_to_file, log_config_to_file
from utils.logger import get_root_logger
from utils import dist_utils, misc
from tools import runner_seg


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type=str, default=None, help='test ckpt path')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.test and args.ckpts is None:
        raise ValueError('--test 需要指定 --ckpts')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name

    if args.resume and not args.test:
        # 断点续训：复用最新一次实验目录，保证能加载到 ckpt-last.pth
        base_dir = os.path.join('./experiments', Path(args.config).stem,
                                Path(args.config).parent.stem, args.exp_name)
        ckpts = sorted(glob.glob(os.path.join(base_dir, '*', 'ckpt-last.pth')))
        if not ckpts:
            raise FileNotFoundError(f'未找到可续训的 ckpt-last.pth，搜索目录: {base_dir}')
        exp_dir = os.path.dirname(ckpts[-1])
        args.experiment_path = exp_dir
        args.tfboard_path = os.path.join('./experiments', Path(args.config).stem,
                                         Path(args.config).parent.stem, 'TFBoard',
                                         args.exp_name, os.path.basename(exp_dir))
    else:
        args.experiment_path = os.path.join(
            './experiments', Path(args.config).stem, Path(args.config).parent.stem,
            args.exp_name, ('%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        args.tfboard_path = os.path.join(
            './experiments', Path(args.config).stem, Path(args.config).parent.stem, 'TFBoard',
            args.exp_name, ('%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    args.log_name = Path(args.config).stem
    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.tfboard_path, exist_ok=True)
    return args


def main():
    args = get_args()
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    if not args.test:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    else:
        train_writer = val_writer = None

    config = get_config(args, logger=logger)

    # set batch size
    if config.dataset.get('train'):
        config.dataset.train.others.bs = config.total_bs
    if config.dataset.get('val'):
        config.dataset.val.others.bs = config.total_bs
    if config.dataset.get('test'):
        config.dataset.test.others.bs = config.total_bs
    # 多任务：语义 + 材质 各自 train/val
    for key in ['sem_train', 'sem_val', 'mat_train', 'mat_val']:
        if config.dataset.get(key):
            config.dataset[key].others.bs = config.total_bs

    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    if args.seed is not None:
        misc.set_random_seed(args.seed, deterministic=False)

    if args.test:
        runner_seg.test_net(args, config)
    elif config.dataset.get('sem_train') is not None:
        runner_seg.run_net_multitask(args, config, train_writer, val_writer)
    else:
        runner_seg.run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    seed_torch(0)
    main()
