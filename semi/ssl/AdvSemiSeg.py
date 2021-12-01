#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 下午3:30
# @Author  : chenxb
# @FileName: AdvSemiSeg.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com
import os
import time
import numpy as np
import pickle
from collections import deque
import shutil

import paddle
import paddle.nn.functional as F

from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, op_flops_funs)
from paddleseg.core.val import evaluate

from semi.models.discriminator import FCDiscriminator
from semi.utils.sampler import SubsetBatchSampler
from semi.losses.advsemiloss import BCEWithLogitsLoss2d

NUM_CLASSES = 21  # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes
SPLIT_ID = None

MODEL = 'DeepLabv2'
NUM_STEPS = 20000
BATCH_SIZE = 10
NUM_WORKERS = 4
SAVE_PRED_EVERY = 5000

INPUT_SIZE = '321,321'
IGNORE_LABEL = 255

RANDOM_SEED = 1234

# DATA_DIRECTORY = '/home/aistudio/data/data4379/pascalvoc/VOCdevkit/VOC2012'
# DATA_LIST_PATH = './data/voc_list/train_aug.txt'
# save_dir = './checkpoints/voc_semi_0_125/'

LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4

MOMENTUM = 0.9
POWER = 0.9
SAVE_NUM_IMAGES = 2
WEIGHT_DECAY = 0.0005
LAMBDA_ADV_PRED = 0.1

SEMI_START=5000
LAMBDA_SEMI=0.1
MASK_T=0.2

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=True

restore_from_D = None

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))

def loss_computation(logits_list, labels, losses, edges=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        # Whether to use edges as labels According to loss type.
        if loss_i.__class__.__name__ in ('BCELoss',
                                         'FocalLoss') and loss_i.edge_label:
            loss_list.append(losses['coef'][i] * loss_i(logits, edges))
        elif loss_i.__class__.__name__ in ("KLLoss", ):
            loss_list.append(losses['coef'][i] * loss_i(
                logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list

def one_hot(label,num_classes):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:, i, ...] = (label == i)
    # handle ignore labels
    return paddle.to_tensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = paddle.to_tensor(D_label).astype('float32')
    return D_label

def train_advSemiSeg(
            cfg,
            model,
            train_dataset,
            label_ratio,
            split_id,
            val_dataset,
            optimizer,
            save_dir,
            iters,
            batch_size,
            resume_model,
            save_interval,
            log_iters,
            num_workers,
            use_vdl,
            losses,
            keep_checkpoint_max,
            test_config,
            fp16,
            profiler_options,
            to_static_training
            ):
    '''
    AdvSemiSeg's train function.
    Returns:
    '''
    # 环境构建
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    # 模型构建
    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    model_D = FCDiscriminator(num_classes=train_dataset.num_classes)
    if restore_from_D is not None:
        model_D.load_state_dict(paddle.load(restore_from_D))

    model.train()
    model_D.train()

    # optimizer for discriminator network
    learning_rate_D = paddle.optimizer.lr.PolynomialDecay(LEARNING_RATE_D, decay_steps=NUM_STEPS,
                                                          power=POWER)
    optimizer_D = paddle.optimizer.Adam(learning_rate=learning_rate_D, parameters=model_D.parameters(), beta1=0.9, beta2=0.99)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)
        optimizer_D = paddle.distributed.fleet.distributed_optimizer(
            optimizer_D)  # The return is Fleet object
        ddp_model_D = paddle.distributed.fleet.distributed_model(model_D)

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    # 数据准备
    if label_ratio is None:
        trainloader = paddle.io.DataLoader(train_dataset,
                                           batch_size=batch_size, num_workers=4)

        trainloader_gt = paddle.io.DataLoader(train_dataset,
                                              batch_size=batch_size, num_workers=4)

        trainloader_remain = paddle.io.DataLoader(train_dataset,
                                                  batch_size=batch_size, num_workers=4)
        trainloader_remain_iter = iter(trainloader_remain)
    else:
        # sample partial data
        train_dataset_size = len(train_dataset)
        partial_size = int(label_ratio * train_dataset_size)

        if split_id is not None:
            train_ids = pickle.load(open(split_id, 'rb'))
            print('loading train ids from {}'.format(split_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(save_dir, 'train_id.pkl'), 'wb'))

        train_sampler = SubsetBatchSampler(indices=train_ids[:partial_size], batch_size=batch_size,
                                           drop_last=True)
        train_remain_sampler = SubsetBatchSampler(indices=train_ids[partial_size:], batch_size=batch_size,
                                                  drop_last=True)
        train_gt_sampler = SubsetBatchSampler(indices=train_ids[:partial_size], batch_size=batch_size,
                                              drop_last=True)

        trainloader = paddle.io.DataLoader(train_dataset,
                                           batch_sampler=train_sampler, num_workers=num_workers,worker_init_fn=worker_init_fn)
        trainloader_remain = paddle.io.DataLoader(train_dataset,
                                                  batch_sampler=train_remain_sampler, num_workers=num_workers,worker_init_fn=worker_init_fn)
        trainloader_gt = paddle.io.DataLoader(train_dataset,
                                              batch_sampler=train_gt_sampler, num_workers=num_workers,worker_init_fn=worker_init_fn)
        trainloader_remain_iter = iter(trainloader_remain)

    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)
    # # use amp
    # if fp16:
    #     logger.info('use amp to train')
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    # 训练
    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    iters_per_epoch = len(train_sampler)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    i_iter = start_iter
    while i_iter < iters:
        i_iter += 1
        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.clear_grad()
        optimizer_D.clear_grad()

        # train G

        # don't accumulate grads in D
        for param in model_D.parameters():
            param.stop_gradient = True

        # do semi first
        if (LAMBDA_SEMI > 0 or LAMBDA_SEMI_ADV > 0) and i_iter >= SEMI_START_ADV:
            try:
                batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)
            reader_cost_averager.record(time.time() - batch_start)

            # only access to img
            images_remain = batch_remain[0]
            interp = paddle.nn.Upsample(size=(images_remain.shape[-2], images_remain.shape[-1]), mode='bilinear', align_corners=True)
            images_remain = paddle.to_tensor(images_remain)
            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                images_remain = images_remain.transpose((0, 2, 3, 1))

            if nranks > 1:
                pred = interp(ddp_model(images_remain)[0])
            else:
                pred = interp(model(images_remain)[0])

            pred_remain = pred.detach()

            if nranks > 1:
                D_out = interp(ddp_model_D(F.softmax(pred, axis=1)))  # 0~1
            else:
                D_out = interp(model_D(F.softmax(pred, axis=1)))  # 0~1

            D_out_sigmoid = F.sigmoid(D_out).cpu().numpy().squeeze(axis=1)
            ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

            # D_label = make_D_label(gt_label, ignore_mask_remain) # 1
            loss_semi_adv = LAMBDA_SEMI_ADV * bce_loss(paddle.cast(D_out, paddle.float32),
                                                            make_D_label(gt_label, ignore_mask_remain))
            # loss_semi_adv.backward()
            loss_semi_adv_value += loss_semi_adv.cpu().numpy()[0] / LAMBDA_SEMI_ADV

            if LAMBDA_SEMI <= 0 or i_iter < SEMI_START:
                loss_semi_adv.backward()
                loss_semi_value = 0
            else:
                # produce ignore mask
                semi_ignore_mask = (D_out_sigmoid < MASK_T)  # TODO fix

                semi_gt = pred.cpu().numpy().argmax(axis=1)
                semi_gt[semi_ignore_mask] = 255

                semi_ratio = 1.0 - float(semi_ignore_mask.sum()) / semi_ignore_mask.size
                print('semi ratio: {:.4f}'.format(semi_ratio))

                if semi_ratio == 0.0:
                    loss_semi_value += 0
                else:
                    semi_gt = paddle.to_tensor(semi_gt)

                    loss_semi = LAMBDA_SEMI * sum(loss_computation([pred], semi_gt,losses))
                    loss_semi_value += loss_semi.cpu().numpy()[0] / LAMBDA_SEMI
                    loss_semi += loss_semi_adv
                    loss_semi.backward()

        else:
            loss_semi = None
            loss_semi_adv = None

        # train with labels

        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels = batch[0], batch[1].astype('int64')
        images = paddle.to_tensor(images)
        if hasattr(model, 'data_format') and model.data_format == 'NHWC':
            images = images.transpose((0, 2, 3, 1))
        ignore_mask = (labels.numpy() == 255)
        if nranks > 1:
            pred = interp(ddp_model(images)[0])
        else:
            pred = interp(model(images)[0])
        loss_seg = sum(loss_computation([pred], labels,losses))  # Cross entropy loss for labeled data
        if nranks > 1:
            D_out = interp(ddp_model_D(F.softmax(pred, axis=1))).astype(paddle.float32)
        else:
            D_out = interp(model_D(F.softmax(pred, axis=1))).astype(paddle.float32)
        # D_out = interp(model_D(F.softmax(pred, axis=1))).astype(paddle.float32)

        loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

        loss = loss_seg + LAMBDA_ADV_PRED * loss_adv_pred

        # proper normalization
        loss.backward()
        loss_seg_value += loss_seg.cpu().numpy()[0]
        loss_adv_pred_value += loss_adv_pred.cpu().numpy()[0]

        # train D

        # bring back requires_grad
        for param in model_D.parameters():
            param.stop_gradient = False

        # train with pred
        pred = pred.detach()

        if D_REMAIN:
            pred = paddle.concat((pred, pred_remain), axis=0)
            ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis=0)

        if nranks > 1:
            D_out = interp(ddp_model_D(F.softmax(pred, axis=1))).astype(paddle.float32)
        else:
            D_out = interp(model_D(F.softmax(pred, axis=1))).astype(paddle.float32)

        loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value += loss_D.cpu().numpy()[0]

        # train with gt
        # get gt labels
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)

        images_gt, labels_gt = batch_gt[0],batch_gt[1]
        D_gt_v = paddle.to_tensor(one_hot(labels_gt,train_dataset.num_classes)).astype('float32')
        ignore_mask_gt = (labels_gt.numpy() == 255)

        if nranks>1:
            D_out = interp(ddp_model_D(D_gt_v)).astype(paddle.float32)
        else:
            D_out = interp(model_D(D_gt_v)).astype(paddle.float32)

        loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value += loss_D.cpu().numpy()[0]

        optimizer.step()
        optimizer_D.step()
        lr = optimizer.get_lr()
        lr_D = optimizer_D.get_lr()
        # update lr
        if isinstance(optimizer, paddle.distributed.fleet.Fleet):
            lr_sche = optimizer.user_defined_optimizer._learning_rate
        else:
            lr_sche = optimizer._learning_rate
        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
            lr_sche.step()
        # lr_sche.step()
        if isinstance(optimizer_D, paddle.distributed.fleet.Fleet):
            lr_sche_D = optimizer_D.user_defined_optimizer._learning_rate
        else:
            lr_sche_D = optimizer_D._learning_rate
        if isinstance(lr_sche_D, paddle.optimizer.lr.LRScheduler):
            lr_sche_D.step()

        batch_cost_averager.record(
            time.time() - batch_start, num_samples=batch_size)
        if (i_iter) % log_iters == 0 and local_rank == 0:
            remain_iters = iters - i_iter
            avg_train_batch_cost = batch_cost_averager.get_average()
            avg_train_reader_cost = reader_cost_averager.get_average()
            eta = calculate_eta(remain_iters, avg_train_batch_cost)
            logger.info(
                "[TRAIN] epoch: {}, iter: {}/{}, loss_seg_value: {:.4f},"
                "loss_adv_pred_value: {:.4f},loss_D_value: {:.4f},"
                "loss_semi_value: {:.4f},loss_semi_adv_value: {:.4f},"
                " lr: {:.6f},lr_D: {:.6f} batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                .format((i_iter - 1) // iters_per_epoch + 1, i_iter, iters,
                            loss_seg_value,loss_adv_pred_value,loss_D_value,
                            loss_semi_value,loss_semi_adv_value, lr, lr_D, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
            if use_vdl:
                log_writer.add_scalar('Train/loss_seg_value', loss_seg_value, i_iter)
                log_writer.add_scalar('Train/loss_adv_pred_value', loss_adv_pred_value, i_iter)
                log_writer.add_scalar('Train/loss_D_value', loss_D_value, i_iter)
                log_writer.add_scalar('Train/loss_semi_value', loss_semi_value, i_iter)
                log_writer.add_scalar('Train/loss_semi_adv_value', loss_semi_adv_value, i_iter)
                log_writer.add_scalar('Train/lr', lr, i_iter)
                log_writer.add_scalar('Train/lr_D', lr_D, i_iter)
                log_writer.add_scalar('Train/batch_cost',
                                      avg_train_batch_cost, i_iter)
                log_writer.add_scalar('Train/reader_cost',
                                      avg_train_reader_cost, i_iter)
        reader_cost_averager.reset()
        batch_cost_averager.reset()
        # 评估
        if (i_iter % save_interval == 0
            or i_iter == iters) and (val_dataset is not None):
            num_workers = 1 if num_workers > 0 else 0

            if test_config is None:
                test_config = {}

            mean_iou, acc, _, _, _ = evaluate(
                model, val_dataset, num_workers=num_workers, **test_config)

            model.train()

        if (i_iter % save_interval == 0 or i_iter == iters) and local_rank == 0:
            current_save_dir = os.path.join(save_dir,
                                            "iter_{}".format(i_iter))
            if not os.path.isdir(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(model_D.state_dict(), os.path.join(current_save_dir,'model_D.pdparams')) # add
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
            paddle.save(optimizer_D.state_dict(),
                        os.path.join(current_save_dir, 'model_D.pdopt'))
            save_models.append(current_save_dir)
            if len(save_models) > keep_checkpoint_max > 0:
                model_to_remove = save_models.popleft()
                shutil.rmtree(model_to_remove)

            if val_dataset is not None:
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    best_model_iter = i_iter
                    best_model_dir = os.path.join(save_dir, "best_model")
                    paddle.save(
                        model.state_dict(),
                        os.path.join(best_model_dir, 'model.pdparams'))
                logger.info(
                    '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                    .format(best_mean_iou, best_model_iter))

                if use_vdl:
                    log_writer.add_scalar('Evaluate/mIoU', mean_iou, i_iter)
                    log_writer.add_scalar('Evaluate/Acc', acc, i_iter)
        batch_start = time.time()

    # Calculate flops.
    if local_rank == 0:
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()