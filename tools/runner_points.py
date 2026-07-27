import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import os
import json
from PIL import Image

from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------
# Compatibility wrapper for backbone loss return values.
# ---------------------------
def _call_get_loss(model_for_loss, rets, gt, gt_missing=None):
    """
    兼容两种 get_loss 签名：
      1) get_loss(rets, gt)
      2) get_loss(rets, gt, gt_missing=gt_missing)

    同时兼容返回：
      - 旧: (loss_total, coarse_loss, final_loss, coarse_loss, final_loss)
      - 新: (loss_total, coarse_loss, final_loss, l_mr,       final_loss)
      - 其他长度：尽量兜底
    """
    try:
        out = model_for_loss.get_loss(rets, gt, gt_missing=gt_missing)
    except TypeError:
        out = model_for_loss.get_loss(rets, gt)

    if not isinstance(out, (tuple, list)):
        raise RuntimeError(f"get_loss must return tuple/list, got {type(out)}")

    # 兜底解析
    loss_total = out[0]
    coarse_loss = out[1] if len(out) > 1 else loss_total
    final_loss  = out[2] if len(out) > 2 else loss_total

    # The fourth item is the missing-region loss for prototype-guided models.
    if len(out) > 3:
        mr_loss = out[3]
    else:
        mr_loss = loss_total.new_zeros(())

    tail = out[4] if len(out) > 4 else final_loss
    return loss_total, coarse_loss, final_loss, mr_loss, tail


def _select_prototype(missing, gt_missing, config):
    """Choose the D2P input while keeping MR supervision tied to GT missing points."""
    source = str(getattr(config, 'prototype_source', 'vgp')).lower()
    if source in {'vgp', 'vcp'}:
        return missing
    if source in ('none', 'no_prototype'):
        # Plain geometry-only backbones must not receive a proxy prior.
        return None
    if source == 'oracle':
        return gt_missing
    if source == 'noise':
        return torch.randn_like(missing)
    raise ValueError(f'Unsupported prototype_source: {source}')


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # -------- Dataset --------
    (train_sampler, train_dataloader), (_, val_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)

    # -------- Model ----------
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # -------- Resume / Init ---
    start_epoch = 0
    best_metrics = None
    metrics = None

    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # -------- DDP / DP --------
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)

        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )
        print_log('Using Distributed Data Parallel ...', logger=logger)
    else:
        print_log('Using Data Parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # -------- Opt / Sched -----
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # val/test 用（全局 CD）
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    base_model.zero_grad()

    # ``max_epoch`` is a count, not an inclusive epoch index.
    for epoch in range(start_epoch, config.max_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Training logs: coarse, final, missing-region, and total losses.
        losses = AverageMeter(['SparseLoss', 'DenseLoss', 'MRLoss', 'Total'])

        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data, missing, gt_missing) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)

            partial = data[0].cuda(non_blocking=True)   # (B,N,3)
            gt      = data[1].cuda(non_blocking=True)   # (B,M,3)

            missing = missing.cuda(non_blocking=True)
            gt_missing = gt_missing.cuda(non_blocking=True)
            prototype = _select_prototype(missing, gt_missing, config)

            num_iter += 1

            rets = base_model(partial, prototype)
            model_for_loss = base_model.module if hasattr(base_model, 'module') else base_model

            loss_total, coarse_loss, final_loss, mr_loss, _ = _call_get_loss(
                model_for_loss, rets, gt, gt_missing=gt_missing
            )

            loss_total.backward()

            # Gradient accumulation.
            if num_iter == config.step_per_update:
                num_iter = 0
                grad_clip = getattr(config, 'grad_clip', None)
                if grad_clip is not None:
                    # PointNet++/transformer decoding in SnowflakeNet can
                    # produce rare large updates on this small dental set.
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), float(grad_clip))
                optimizer.step()
                base_model.zero_grad()

            # Aggregate metrics in distributed runs.
            if args.distributed:
                coarse_loss = dist_utils.reduce_tensor(coarse_loss, args)
                final_loss  = dist_utils.reduce_tensor(final_loss,  args)
                mr_loss     = dist_utils.reduce_tensor(mr_loss,     args)
                loss_total  = dist_utils.reduce_tensor(loss_total,  args)

            # Keep logged distances in the paper's x1e3 scale.
            losses.update([
                coarse_loss.item() * 1000.0,
                final_loss.item()  * 1000.0,
                mr_loss.item()     * 1000.0,
                loss_total.item()  * 1000.0
            ])

            if args.distributed:
                torch.cuda.synchronize()

            # TensorBoard.
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', coarse_loss.item() * 1000.0, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense',  final_loss.item()  * 1000.0, n_itr)
                train_writer.add_scalar('Loss/Batch/MR',     mr_loss.item()     * 1000.0, n_itr)
                train_writer.add_scalar('Loss/Batch/Total',  loss_total.item()  * 1000.0, n_itr)
                train_writer.add_scalar('LR/training', optimizer.param_groups[0]['lr'], n_itr)

            # Logging.
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            if idx % args.print_freq == 0:
                print_log(
                    '[Epoch %d/%d][Batch %d/%d] '
                    'BatchTime=%.3f(s) DataTime=%.3f(s) Losses=%s lr=%.6f' %
                    (epoch, config.max_epoch, idx + 1, n_batches,
                     batch_time.val(), data_time.val(),
                     ['%.4f' % l for l in losses.val()],
                     optimizer.param_groups[0]['lr']),
                    logger=logger
                )

        # Scheduler step
        if isinstance(scheduler, list):
            for sch in scheduler:
                sch.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_end_time = time.time()

        # Epoch-level tb
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense',  losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/MR',     losses.avg(2), epoch)
            train_writer.add_scalar('Loss/Epoch/Total',  losses.avg(3), epoch)

        print_log(
            '[Training] EPOCH: %d EpochTime=%.3f(s) Losses=%s' %
            (epoch, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in losses.avg()]),
            logger=logger
        )

        # Validation and checkpointing.
        if (epoch % args.val_freq == 0 and epoch != 0) or ((config.max_epoch - epoch) < 30):
            metrics = validate(
                base_model, val_dataloader, epoch,
                ChamferDisL1, ChamferDisL2,
                val_writer, args, config, logger=logger
            )
            if best_metrics is None or metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch,
                                        metrics, best_metrics, 'ckpt-best', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch,
                                metrics, best_metrics, 'ckpt-last', args, logger=logger)

        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch,
                                    metrics, best_metrics,
                                    f'ckpt-epoch-{epoch:03d}', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, val_dataloader, epoch,
             ChamferDisL1, ChamferDisL2,
             val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()

    # Loss logs: coarse, final, missing-region, and total.
    val_losses = AverageMeter(['SparseLossL1', 'DenseLossL1', 'CD_Miss', 'Total'])

    # Metrics: F-Score, CDL1, CDL2, and CDMiss.
    val_metrics = AverageMeter(Metrics.names_with_missing())
    category_metrics = dict()
    n_samples = len(val_dataloader)

    # CDMiss settings.
    k_pred = getattr(config, 'k_pred_miss', 10)
    squared = True

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, missing, gt_missing) in enumerate(val_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            partial = data[0].cuda(non_blocking=True)
            gt      = data[1].cuda(non_blocking=True)

            missing = missing.cuda(non_blocking=True)
            gt_missing = gt_missing.cuda(non_blocking=True)
            prototype = _select_prototype(missing, gt_missing, config)

            rets = base_model(partial, prototype)
            coarse_points = rets[0]
            dense_points  = rets[-1]

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            dense_loss_l1  = ChamferDisL1(dense_points,  gt)

            model_for_loss = base_model.module if hasattr(base_model, 'module') else base_model
            loss_total, _, _, _, _ = _call_get_loss(
                model_for_loss, rets, gt, gt_missing=gt_missing
            )

            _metrics = Metrics.get_with_gt_missing(
                dense_points, gt, gt_missing,
                k_pred=k_pred, squared=squared
            )
            cd_miss = _metrics[-1]  # *1000 的 float

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                dense_loss_l1  = dist_utils.reduce_tensor(dense_loss_l1,  args)
                loss_total     = dist_utils.reduce_tensor(loss_total,     args)

                cd_miss_t = dense_points.new_tensor(cd_miss / 1000.0)
                cd_miss_t = dist_utils.reduce_tensor(cd_miss_t, args)
                cd_miss = cd_miss_t.item() * 1000.0

            val_losses.update([
                sparse_loss_l1.item() * 1000.0,
                dense_loss_l1.item()  * 1000.0,
                cd_miss,
                loss_total.item()     * 1000.0
            ])

            val_metrics.update(_metrics)
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names_with_missing())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % args.val_interval == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_img = misc.get_ptcloud_img(input_pc)
                val_writer.add_image(f'Val{idx:02d}-{epoch}/Input', input_img, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image(f'Val{idx:02d}-{epoch}/Sparse', sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image(f'Val{idx:02d}-{epoch}/Dense', dense_img, epoch, dataformats='HWC')

                gt_np = gt.squeeze().cpu().numpy()
                gt_img = misc.get_ptcloud_img(gt_np)
                val_writer.add_image(f'Val{idx:02d}-{epoch}/GT', gt_img, epoch, dataformats='HWC')

                print_log(
                    'Validation[%d/%d] Taxonomy=%s Sample=%s Losses=%s Metrics=%s' %
                    (idx, n_samples, taxonomy_id, model_id,
                     ['%.4f' % l for l in val_losses.val()],
                     ['%.4f' % m for m in _metrics]),
                    logger=logger
                )

        for _, v in category_metrics.items():
            val_metrics.update(v.avg())

        print_log(
            '[Validation] EPOCH: %d  Metrics = %s' %
            (epoch, ['%.4f' % m for m in val_metrics.avg()]),
            logger=logger
        )

        if args.distributed:
            torch.cuda.synchronize()

    shapenet_dict = json.load(open(
        'data/tooth_synset_dict.json', 'r'
    ))
    print_log('============================ VAL RESULTS ============================', logger=logger)
    msg = 'Taxonomy\t#Sample\t'
    for metric in val_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = taxonomy_id + '\t'
        msg += str(category_metrics[taxonomy_id].count(0)) + '\t'
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict.get(taxonomy_id, taxonomy_id) + '\t'
        print_log(msg, logger=logger)

    msg = 'Overall\t\t'
    for value in val_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/SparseL1', val_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/DenseL1',  val_losses.avg(1), epoch)
        val_writer.add_scalar('Loss/Epoch/CD_Miss',  val_losses.avg(2), epoch)
        val_writer.add_scalar('Loss/Epoch/Total',    val_losses.avg(3), epoch)
        for i, metric in enumerate(val_metrics.items):
            val_writer.add_scalar(f'Metric/{metric}', val_metrics.avg(i), epoch)

    base_len = len(Metrics.names())
    return Metrics(config.consider_metric, val_metrics.avg()[:base_len])


# ===================== Test（牙齿 + 缺失区域 MR 指标） =====================

def test_net(args, config, test_writer=None):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    print_log(base_model, logger=logger)

    state_dict = torch.load(args.ckpts, map_location='cpu')['base_model']
    weights_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    base_model.load_state_dict(weights_dict)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, test_writer, args, config, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2,
         test_writer, args, config, logger=None):

    base_model.eval()

    test_losses = AverageMeter([
        'SparseLossL1', 'SparseLossL2',
        'DenseLossL1',  'DenseLossL2',
        'CD_Miss_Coarse', 'CD_Miss_Dense',
        'Total'
    ])

    test_metrics = AverageMeter(Metrics.names_with_missing())
    category_metrics = dict()

    n_samples = len(test_dataloader)
    print(f"n_samples: {n_samples}")

    k_pred = getattr(config, 'k_pred_miss', 10)
    squared = True

    save_dir = os.path.join("inference_result", config.model.NAME, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, missing, gt_missing) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            partial = data[0].cuda(non_blocking=True)
            gt      = data[1].cuda(non_blocking=True)

            missing = missing.cuda(non_blocking=True)
            gt_missing = gt_missing.cuda(non_blocking=True)
            prototype = _select_prototype(missing, gt_missing, config)

            rets = base_model(partial, prototype)
            coarse_points = rets[0]
            dense_points  = rets[-1]

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
            dense_loss_l1  = ChamferDisL1(dense_points,  gt)
            dense_loss_l2  = ChamferDisL2(dense_points,  gt)

            model_for_loss = base_model.module if hasattr(base_model, 'module') else base_model
            loss_total, _, _, _, _ = _call_get_loss(
                model_for_loss, rets, gt, gt_missing=gt_missing
            )

            miss_coarse = Metrics.get_with_gt_missing(
                coarse_points, gt, gt_missing, k_pred=k_pred, squared=squared
            )[-1]
            miss_dense = Metrics.get_with_gt_missing(
                dense_points,  gt, gt_missing, k_pred=k_pred, squared=squared
            )[-1]

            test_losses.update([
                sparse_loss_l1.item() * 1000.0,
                sparse_loss_l2.item() * 1000.0,
                dense_loss_l1.item()  * 1000.0,
                dense_loss_l2.item()  * 1000.0,
                miss_coarse,
                miss_dense,
                loss_total.item()     * 1000.0
            ])

            _metrics = Metrics.get_with_gt_missing(
                dense_points, gt, gt_missing, k_pred=k_pred, squared=squared
            )
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names_with_missing())
            category_metrics[taxonomy_id].update(_metrics)

            B = dense_points.shape[0]
            for i in range(B):
                dense_np = dense_points[i].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(dense_np)
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{model_ids[i]}.pcd"), pcd)

            print_log(
                'Test[%d/%d] Taxonomy=%s Sample=%s Losses=%s Metrics=%s' %
                (idx, n_samples, taxonomy_id, model_id,
                 ['%.4f' % l for l in test_losses.val()],
                 ['%.4f' % m for m in _metrics]),
                logger=logger
            )

        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' %
                  (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

    shapenet_dict = json.load(open(
        'data/tooth_synset_dict.json', 'r'
    ))
    print_log('============================ TEST RESULTS ============================', logger=logger)

    msg = 'Taxonomy\t#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = taxonomy_id + '\t'
        msg += str(category_metrics[taxonomy_id].count(0)) + '\t'
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict.get(taxonomy_id, taxonomy_id) + '\t'
        print_log(msg, logger=logger)

    msg = 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Preserve the full-precision, x1e3-scaled values used by the printed
    # table. Queue and Pareto tooling consume this sidecar instead of parsing
    # the rounded human-readable log line.
    metric_payload = {
        'metric_scale': 'x1e3',
        'k_pred_miss': int(k_pred),
        'metrics': {
            name: float(value)
            for name, value in zip(Metrics.names_with_missing(), test_metrics.avg())
        },
    }
    with open(os.path.join(args.experiment_path, 'test_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metric_payload, handle, indent=2)

    return
