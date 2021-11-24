import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from model.discriminator import FCDiscriminator
from data.voc_dataset import VOCDataSet
from data import get_loader, get_data_path
from data.augmentations import *

from data.sampler import SubsetRandomSampler
from paddleseg.utils import get_sys_env, logger
from paddleseg.models.backbones import resnet_vd
from model.deeplabv2 import DeepLabV2
import paddleseg.transforms as transform
from data.voc_dataset import VOCDataSet
from data import get_data_path, get_loader

from PIL import Image
import scipy.misc


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'pascal_voc' # pascal_context

MODEL = 'Deeplabv2' # deeeplabv2, deeplabv3p
DATA_DIRECTORY = '/home/aistudio/data/data4379/pascalvoc/VOCdevkit/VOC2012'
DATA_LIST_PATH = './data/voc_list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21 # 60 for pascal context
RESTORE_FROM = ''
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset name pascal_voc or pascal_context")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = paddle.to_tensor(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(args, data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if args.dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif args.dataset == 'pascal_context':
        classes = np.array(('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench', 'book', 'building', 'cabinet' , 'ceiling', 'cloth', 'computer', 'cup',
                'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate',
                'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'))
    elif args.dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "terrain", "sky", "person", "rider",
            "car", "truck", "bus",
            "train", "motorcycle", "bicycle")) 

    for i, iou in enumerate(j_list):
        if j_list[i] > 0:
            print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))
    
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    # 初始化设置
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 创建分割网络
    backbone = resnet_vd.ResNet101_vd()
    model = DeepLabV2(num_classes=args.num_classes,backbone=backbone,pretrained=args.restore_from)

    # saved_state_dict = paddle.load(args.restore_from)
    # model.load_state_dict(saved_state_dict)

    model.eval()

    if args.dataset == 'pascal_voc':
        testloader = paddle.io.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False)
        interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)

    elif args.dataset == 'pascal_context':
        input_transform = Compose([
                transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        data_kwargs = {'transform': input_transform, 'base_size': 512, 'crop_size': 512}
        data_loader = get_loader('pascal_context')
        data_path = get_data_path('pascal_context')
        test_dataset = data_loader(data_path, split='val', mode='val', **data_kwargs)
        testloader = paddle.io.DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=1)
        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    elif args.dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader( data_path, img_size=(512, 1024), is_transform=True, split='val')
        testloader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False)
        interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
    
    data_list = []
    colorize = VOCColorize()
 
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name, _ = batch
        size = size[0]
        output  = model(paddle.to_tensor(image))[0].squeeze(0).cpu().numpy()
        if args.dataset == 'pascal_voc':
            output = output[:,:size[0],:size[1]]
            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        elif args.dataset == 'pascal_context':
            gt = np.asarray(label[0].numpy(), dtype=np.int)
        elif args.dataset == 'cityscapes':
            gt = np.asarray(label[0].numpy(), dtype=np.int)

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
       
        if args.save_output_images:
            if args.dataset == 'pascal_voc':
                filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
                color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(filename)
            elif args.dataset == 'pascal_context':
                filename = os.path.join(args.save_dir, filename[0])
                scipy.misc.imsave(filename, gt)
        
        data_list.append([gt.flatten(), output.flatten()])
    
    filename = os.path.join(args.save_dir, 'result.txt')
    get_iou(args, data_list, args.num_classes, filename)


if __name__ == '__main__':
    main()
