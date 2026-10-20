"""PointMambaSeg 训练/验证/测试 runner（逐点语义分割，mIoU 指标）。"""
import os
import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import numpy as np


class Seg_Metric:
    def __init__(self, miou=0., oa=0.):
        self.miou = miou
        self.oa = oa

    def better_than(self, other):
        return self.miou > other.miou

    def state_dict(self):
        return {'miou': self.miou, 'oa': self.oa}


def compute_miou(conf_matrix):
    """从混淆矩阵计算 mIoU 和 overall accuracy。

    conf_matrix: (C, C) int64 tensor, conf[i][j] = GT i 被预测为 j 的数量。
    """
    C = conf_matrix.shape[0]
    iou = torch.zeros(C, device=conf_matrix.device)
    for c in range(C):
        gt = conf_matrix[c].sum()      # GT 类别 c 总数
        pred = conf_matrix[:, c].sum() # 预测为 c 总数
        tp = conf_matrix[c, c]
        union = gt + pred - tp
        iou[c] = (tp.float() / union.float()) if union > 0 else float('nan')

    valid_iou = iou[~torch.isnan(iou)]
    miou = valid_iou.mean().item() if valid_iou.numel() > 0 else 0.0
    oa = conf_matrix.diag().sum().float() / conf_matrix.sum().float()
    return miou, oa.item(), iou.cpu().numpy()


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    num_classes = config.model.num_classes

    # ---- dataset ----
    (train_sampler, train_dataloader), (_, val_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)

    # ---- model ----
    base_model = builder.model_builder(config.model)

    start_epoch = 0
    best_metrics = Seg_Metric(0.)

    if args.resume:
        start_epoch, bm = builder.resume_model(base_model, args, logger=logger)
        # 必须恢复 best_metrics：初值恒为 0，续训后第一次验证必然 better_than(0)
        # 成立，会无条件覆盖 ckpt-best.pth，把崩溃前的最优权重冲掉。
        # resume_model 在没有 ckpt 时返回 0，此时保持初值。
        if bm:
            best_metrics = Seg_Metric(bm.get('miou', 0.), bm.get('oa', 0.))
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
    else:
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    base_model.zero_grad()
    misc.summary_parameters(base_model, logger=logger)

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        losses = AverageMeter(['loss'])
        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            points = data[0].cuda()
            sem = data[1].cuda()

            ret, _ = base_model(points)  # (B,N,C), (B,N,M)|None
            loss = base_model.module.get_loss(ret, sem)

            loss.backward()
            num_iter += 1
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item()])

            if train_writer is not None:
                n_itr = epoch * n_batches + idx
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        print_log('[Training] EPOCH: %d  Loss = %.4f  lr = %.6f' %
                  (epoch, losses.avg(0), optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, val_dataloader, num_classes, epoch, val_writer, args, logger=logger)
            print_log('[Validation] EPOCH: %d  mIoU = %.4f  OA = %.4f' % (epoch, metrics.miou, metrics.oa),
                      logger=logger)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, None, best_metrics, 'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    print_log('训练结束，best mIoU = %.4f, best OA = %.4f' % (best_metrics.miou, best_metrics.oa), logger=logger)


def validate(base_model, test_dataloader, num_classes, epoch, val_writer, args, config=None, logger=None, task='sem'):
    base_model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64).cuda()

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            sem = data[1].cuda()

            sem_logits, mat_logits = base_model(points)  # (B,N,C), (B,N,M)|None
            logits = sem_logits if task == 'sem' else mat_logits
            pred = logits.argmax(-1)     # (B, N)

            valid = sem >= 0
            pred_v = pred[valid].long()
            sem_v = sem[valid].long()

            # 累加混淆矩阵
            for c in range(num_classes):
                gt_mask = (sem_v == c)
                if gt_mask.sum() == 0:
                    continue
                pred_c = pred_v[gt_mask]
                conf[c] += torch.bincount(pred_c, minlength=num_classes)

    miou, oa, iou_per_class = compute_miou(conf)

    if val_writer is not None:
        val_writer.add_scalar('Metric/mIoU', miou, epoch)
        val_writer.add_scalar('Metric/OA', oa, epoch)

    return Seg_Metric(miou, oa)


def test_net(args, config):
    logger = get_logger(args.log_name)
    num_classes = config.model.num_classes
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model = nn.DataParallel(base_model).cuda()

    metrics = validate(base_model, test_dataloader, num_classes, 0, None, args, logger=logger)
    print_log('[TEST] mIoU = %.4f, OA = %.4f' % (metrics.miou, metrics.oa), logger=logger)


# --------------------------------------------------------------------------- #
# 多任务训练：共享编码器 + 语义头(nuScenes) + 材质头(UR-MAT)，逐 batch 交替
# --------------------------------------------------------------------------- #
class MultiTaskMetric:
    def __init__(self, sem_miou=0., mat_miou=0.):
        self.sem_miou = sem_miou
        self.mat_miou = mat_miou
        self.score = (sem_miou + mat_miou) / 2.

    def better_than(self, other):
        return self.score > other.score

    def state_dict(self):
        return {'sem_miou': self.sem_miou, 'mat_miou': self.mat_miou,
                'score': self.score}


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def run_net_multitask(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    num_classes = config.model.num_classes
    num_materials = config.model.num_materials

    # ---- 4 个 dataloader：语义(nuScenes) + 材质(UR-MAT) 各自 train/val ----
    (sem_train_sampler, sem_train_loader) = builder.dataset_builder(args, config.dataset.sem_train)
    (_, sem_val_loader) = builder.dataset_builder(args, config.dataset.sem_val)
    (mat_train_sampler, mat_train_loader) = builder.dataset_builder(args, config.dataset.mat_train)
    (_, mat_val_loader) = builder.dataset_builder(args, config.dataset.mat_val)

    # ---- model ----
    base_model = builder.model_builder(config.model)
    start_epoch = 0
    best_metrics = MultiTaskMetric(0., 0.)

    if args.resume:
        start_epoch, bm = builder.resume_model(base_model, args, logger=logger)
        # 同 run_net：不恢复 best_metrics 会在续训首轮冲掉 ckpt-best.pth。
        # MultiTaskMetric.score 由 sem/mat 重算，与 state_dict 里存的一致。
        if bm:
            best_metrics = MultiTaskMetric(bm.get('sem_miou', 0.), bm.get('mat_miou', 0.))
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
    else:
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    base_model.zero_grad()
    misc.summary_parameters(base_model, logger=logger)

    n_sem = len(sem_train_loader)
    n_mat = len(mat_train_loader)
    n_batches = n_sem + n_mat
    n_pairs = min(n_sem, n_mat)   # = n_mat
    # 可选损失加权（默认 1:1，调高 mat_loss_weight 补偿材质更新次数少）
    sem_w = float(getattr(config, 'sem_loss_weight', 1.0))
    mat_w = float(getattr(config, 'mat_loss_weight', 1.0))

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            sem_train_sampler.set_epoch(epoch)
            mat_train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        losses = AverageMeter(['loss'])
        sem_losses = AverageMeter(['sem_loss'])
        mat_losses = AverageMeter(['mat_loss'])
        sem_iter = iter(sem_train_loader)   # 单遍，不 cycle
        mat_iter = iter(mat_train_loader)   # 单遍，不 cycle

        # 聚类调度：材质 batch 集中在前 2*n_pairs 步（与语义 1:1 交替），之后纯语义。
        # 语义任务获得"无材质干扰"的干净阶段（对主任务最有利）；mat_w 补偿材质在
        # 纯语义阶段被覆盖的倾向。
        for idx in range(n_batches):
            if idx < 2 * n_pairs:
                if idx % 2 == 0:
                    _, _, data = next(sem_iter); task = 'sem'
                else:
                    _, _, data = next(mat_iter); task = 'mat'
            else:
                _, _, data = next(sem_iter); task = 'sem'

            points = data[0].cuda()
            label = data[1].cuda()

            sem_logits, mat_logits = base_model(points)
            if task == 'sem':
                loss = sem_w * base_model.module.get_loss(sem_logits, label)
                sem_losses.update([loss.item()])
            else:
                loss = mat_w * base_model.module.get_loss(mat_logits, label)
                mat_losses.update([loss.item()])

            loss.backward()
            if (idx + 1) % config.step_per_update == 0:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item()])
            if train_writer is not None:
                n_itr = epoch * n_batches + idx
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                if task == 'sem':
                    train_writer.add_scalar('Loss/Batch/SemLoss', loss.item(), n_itr)
                else:
                    train_writer.add_scalar('Loss/Batch/MatLoss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        print_log('[Training] EPOCH: %d  Loss = %.4f  sem = %.4f  mat = %.4f  lr = %.6f' %
                  (epoch, losses.avg(0), sem_losses.avg(0), mat_losses.avg(0),
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            sem_metric = validate(base_model, sem_val_loader, num_classes, epoch, val_writer, args, logger=logger, task='sem')
            mat_metric = validate(base_model, mat_val_loader, num_materials, epoch, val_writer, args, logger=logger, task='mat')
            metrics = MultiTaskMetric(sem_metric.miou, mat_metric.miou)
            print_log('[Validation] EPOCH: %d  sem_mIoU = %.4f  mat_mIoU = %.4f  avg = %.4f' %
                      (epoch, metrics.sem_miou, metrics.mat_miou, metrics.score), logger=logger)
            if val_writer is not None:
                val_writer.add_scalar('Metric/sem_mIoU', metrics.sem_miou, epoch)
                val_writer.add_scalar('Metric/mat_mIoU', metrics.mat_miou, epoch)
                val_writer.add_scalar('Metric/avg_mIoU', metrics.score, epoch)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, None, best_metrics, 'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    print_log('训练结束，best sem_mIoU = %.4f, best mat_mIoU = %.4f, best avg = %.4f' %
              (best_metrics.sem_miou, best_metrics.mat_miou, best_metrics.score), logger=logger)
