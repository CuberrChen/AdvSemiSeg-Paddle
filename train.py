import argparse
import os
import time
import cv2
import numpy as np
import pickle
import scipy.misc

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import paddleseg.transforms as transform

from model.discriminator import FCDiscriminator
from data.voc_dataset import VOCDataSet,VOCGTDataSet
from data import get_loader, get_data_path
from data.augmentations import *

from data.sampler import SubsetBatchSampler
from paddleseg.utils import get_sys_env, logger
from paddleseg.models.backbones import resnet_vd
from model.deeplabv2 import DeepLabV2
# from paddleseg.models.losses import CrossEntropyLoss as CrossEntropy2d
from utils.loss import BCEWithLogitsLoss2d,CrossEntropy2d

import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 21  # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes
SPLIT_ID = None

MODEL = 'DeepLabv2'
PRETRAINED_BACKBONE = 'https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld.tar.gz'

NUM_STEPS = 20000
BATCH_SIZE = 10
NUM_WORKERS = 4
SAVE_PRED_EVERY = 5000

INPUT_SIZE = '321,321'
IGNORE_LABEL = 255

RANDOM_SEED = 1234

DATA_DIRECTORY = '/home/aistudio/data/data4379/pascalvoc/VOCdevkit/VOC2012'
DATA_LIST_PATH = './data/voc_list/train_aug.txt'
CHECKPOINT_DIR = './checkpoints/voc_semi_0_125/'

LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4

MOMENTUM = 0.9
POWER = 0.9
SAVE_NUM_IMAGES = 2
WEIGHT_DECAY = 0.0005
LAMBDA_ADV_PRED = 0.1

LABELED_RATIO = 0.125  # 0.02 # 1/8 labeled data by default

SEMI_START=5000
LAMBDA_SEMI=0.1
MASK_T=0.2  

ITER_SIZE=1

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=True


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument('--pretrained_backbone',dest='pretrained_backbone',default=PRETRAINED_BACKBONE,type=str,
        help='Set pretrained_backbone during training.')
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument('--seed',dest='seed',default=RANDOM_SEED,type=int,
        help='Set the random seed during training.')
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="split order id")
    parser.add_argument("--iter_size", type=int, default=ITER_SIZE,
                        help="accumlate iter.")      
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label):
    criterion = CrossEntropy2d()
    return criterion(pred, label)

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:, i, ...] = (label == i)
    # handle ignore labels
    return paddle.to_tensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = paddle.to_tensor(D_label).astype('float32')
    return D_label


def main():
    # 初始化设置
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    print(args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # 创建分割网络
    backbone = resnet_vd.ResNet101_vd(pretrained=args.pretrained_backbone) # 101
    model = DeepLabV2(num_classes=args.num_classes,backbone=backbone)
    model.train()

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(paddle.load(args.restore_from_D))
    model_D.train()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(args.checkpoint_dir)
    
    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    if args.labeled_ratio is None:
        trainloader = paddle.io.DataLoader(train_dataset,
                                      batch_size=args.batch_size, num_workers=4,use_shared_memory=False)

        trainloader_gt = paddle.io.DataLoader(train_gt_dataset,
                                         batch_size=args.batch_size, num_workers=4,use_shared_memory=False)

        trainloader_remain = paddle.io.DataLoader(train_dataset,
                                             batch_size=args.batch_size, num_workers=4,use_shared_memory=False)
        trainloader_remain_iter = iter(trainloader_remain)
    else:
        #sample partial data
        partial_size = int(args.labeled_ratio * train_dataset_size)

        if args.split_id is not None:
            train_ids = pickle.load(open(args.split_id,'rb'))
            print('loading train ids from {}'.format(args.split_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(os.path.join(args.checkpoint_dir, 'train_id.pkl'), 'wb'))

        train_sampler = SubsetBatchSampler(indices=train_ids[:partial_size],batch_size=args.batch_size,drop_last=True)
        train_remain_sampler = SubsetBatchSampler(indices=train_ids[partial_size:],batch_size=args.batch_size,drop_last=True)
        train_gt_sampler = SubsetBatchSampler(indices=train_ids[:partial_size],batch_size=args.batch_size,drop_last=True)

        trainloader = paddle.io.DataLoader(train_dataset,
                         batch_sampler=train_sampler, num_workers=4,use_shared_memory=False)
        trainloader_remain = paddle.io.DataLoader(train_dataset,
                         batch_sampler=train_remain_sampler, num_workers=4,use_shared_memory=False)
        trainloader_gt = paddle.io.DataLoader(train_gt_dataset,
                       batch_sampler=train_gt_sampler, num_workers=4,use_shared_memory=False)

        trainloader_remain_iter = iter(trainloader_remain)


    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)
    print('--------------------------- dataloader construction done! --------------------------')

    print('--------------------------- optimizer construction --------------------------')
    # optimizer for segmentation network
    learning_rate = paddle.optimizer.lr.PolynomialDecay(args.learning_rate,decay_steps=args.num_steps,power=args.power)
    optimizer = optim.Momentum(learning_rate=learning_rate,parameters=model.parameters(), momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.clear_grad()

    # optimizer for discriminator network
    learning_rate_D = paddle.optimizer.lr.PolynomialDecay(args.learning_rate_D,decay_steps=args.num_steps,power=args.power)
    optimizer_D = optim.Adam(learning_rate=learning_rate_D,parameters=model_D.parameters(),beta1=0.9,beta2=0.99)
    optimizer_D.clear_grad()
    print('--------------------------- optimizer construction done! --------------------------')

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)    
    
    # labels for adversarial training
    pred_label = 0
    gt_label = 1


    for i_iter in range(args.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.clear_grad()
        optimizer_D.clear_grad()

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.stop_gradient = True

            # do semi first
            if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and i_iter >= args.semi_start_adv :
                try:
                    batch = next(trainloader_remain_iter)
                except:
                    trainloader_remain_iter = iter(trainloader_remain)
                    batch = next(trainloader_remain_iter)

                # only access to img
                images, _, _, _, _ = batch
                images = paddle.to_tensor(images)

                pred = interp(model(images)[0]) # n*c*h*w
                pred_remain = pred.detach()

                D_out = interp(model_D(F.softmax(pred,axis=1))) # n*1*h*w
    
                D_out_sigmoid = F.sigmoid(D_out).cpu().numpy().squeeze(axis=1) # n*h*w

                ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

                loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
                loss_semi_adv = loss_semi_adv/args.iter_size

                #loss_semi_adv.backward()
                loss_semi_adv_value += loss_semi_adv.cpu().numpy()[0]/args.lambda_semi_adv

                if args.lambda_semi <= 0 or i_iter < args.semi_start:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:
                    # produce ignore mask
                    semi_ignore_mask = (D_out_sigmoid < args.mask_T)

                    semi_gt = pred.cpu().numpy().argmax(axis=1)
                    semi_gt[semi_ignore_mask] = 255

                    semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
                    print('semi ratio: {:.4f}'.format(semi_ratio))

                    if semi_ratio == 0.0:
                        loss_semi_value += 0
                    else:
                        semi_gt = paddle.to_tensor(semi_gt)
                        loss_semi = args.lambda_semi * loss_calc(pred, semi_gt)
                        loss_semi = loss_semi/args.iter_size
                        loss_semi_value += loss_semi.cpu().numpy()[0]/args.lambda_semi
                        loss_semi += loss_semi_adv
                        loss_semi.backward()

            else:
                loss_semi = None
                loss_semi_adv = None

            # train with source

            try:
                batch = next(trainloader_iter)
            except:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)

            images, labels, _, _, _ = batch
            images = paddle.to_tensor(images)
            ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images)[0])

            loss_seg = loss_calc(pred, labels)

            D_out = interp(model_D(F.softmax(pred,axis=1)))

            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.cpu().numpy()[0]/args.iter_size
            loss_adv_pred_value += loss_adv_pred.cpu().numpy()[0]/args.iter_size


            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.stop_gradient = False

            # train with pred
            pred = pred.detach()

            if args.D_remain:
                pred = paddle.concat((pred, pred_remain), 0)
                ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)

            D_out = interp(model_D(F.softmax(pred,axis=1)))
            loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.cpu().numpy()[0]


            # train with gt
            # get gt labels
            try:
                batch = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = iter(trainloader_gt)
                batch = next(trainloader_gt_iter)

            _, labels_gt, _, _ ,_ = batch
            D_gt_v = paddle.to_tensor(one_hot(labels_gt))
            ignore_mask_gt = (labels_gt.numpy() == 255)

            D_out = interp(model_D(D_gt_v))
            loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.cpu().numpy()[0]


        optimizer.step()
        optimizer_D.step()
        lr = optimizer.get_lr()
        lr_D = optimizer_D.get_lr()
        lr_sche = optimizer._learning_rate
        lr_sche_D = optimizer_D._learning_rate
        lr_sche.step()
        lr_sche_D.step()

        print('exp = {}'.format(args.checkpoint_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, '
              'loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(i_iter, args.num_steps,
                                                                    loss_seg_value, loss_adv_pred_value,
                                                                    loss_D_value, loss_semi_value, loss_semi_adv_value))
        if args.use_vdl:
            log_writer.add_scalar('Train/loss_seg_value', loss_seg_value, i_iter)
            log_writer.add_scalar('Train/loss_adv_pred_value', loss_adv_pred_value, i_iter)
            log_writer.add_scalar('Train/loss_D_value', loss_D_value, i_iter)
            log_writer.add_scalar('Train/loss_semi_value', loss_semi_value, i_iter)
            log_writer.add_scalar('Train/loss_semi_adv_value', loss_semi_adv_value, i_iter)
            log_writer.add_scalar('Train/lr', lr, i_iter)
            log_writer.add_scalar('Train/lr_D', lr_D, i_iter)


        if i_iter >= args.num_steps - 1:
            print('save model ...')
            paddle.save(model.state_dict(), os.path.join(args.checkpoint_dir, str(args.num_steps) + '.pdparams'))
            paddle.save(model_D.state_dict(), os.path.join(args.checkpoint_dir, str(args.num_steps) + '_D.pdparams'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('saving checkpoint  ...')
            paddle.save(model.state_dict(), os.path.join(args.checkpoint_dir, str(i_iter) + '.pdparams'))
            paddle.save(model_D.state_dict(), os.path.join(args.checkpoint_dir, str(i_iter) + '_D.pdparams'))

    end = timeit.default_timer()
    time.sleep(0.5)
    if args.use_vdl:
        log_writer.close()
    print(end - start, 'seconds')
    print('--------------------------- train done!--------------------------')


if __name__ == '__main__':
    main()