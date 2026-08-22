# modified from: https://github.com/yinboc/liif

import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD
from tensorboardX import SummaryWriter
from Adam import Adam
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import math

def show_feature_map(feature_map,layer,name='rgb',rgb=False):
    feature_map = feature_map.squeeze(0)
    #if rgb: feature_map = feature_map.permute(1,2,0)*0.5+0.5
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = math.ceil(np.sqrt(feature_map_num))
    if rgb:
        #plt.figure()
        #plt.imshow(feature_map)
        #plt.axis('off')
        feature_map = cv2.cvtColor(feature_map,cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/'+layer+'/'+name+".png",feature_map*255)
        #plt.show()
    else:
        plt.figure()
        for index in range(1, feature_map_num+1):
            t = (feature_map[index-1]*255).astype(np.uint8)
            t = cv2.applyColorMap(t, cv2.COLORMAP_TWILIGHT)
            plt.subplot(row_num, row_num, index)
            plt.imshow(t, cmap='gray')
            plt.axis('off')
            #ensure_path('data/'+layer)
            cv2.imwrite('data/'+layer+'/'+str(name)+'_'+str(index)+".png",t)
        #plt.show()
        plt.savefig('data/'+layer+'/'+str(name)+".png")

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.3f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.3f}m'.format(t / 60)
    else:
        return '{:.3f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path

# 用于打印和记录日志的函数，它的主要功能是将传入的对象 obj 打印到控制台，并且如果全局变量 _log_path 已经设置，会将日志内容保存到文件中。
def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

# remove来判断是否要重新创建新目录  路径存在并以 _ 开头：自动删除并重建 路径存在且不以 _ 开头：询问用户是否删除。
def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    # 路径存在并以 _ 开头：自动删除并重建 路径存在且不以 _ 开头：询问用户是否删除。
    ensure_path(save_path, remove=remove)
    # 日志路径设置为一个全局变量 _log_path
    set_log_path(save_path)
    # 会在 save_path 下创建一个 tensorboard 文件夹，用于存储 TensorBoard 相关的日志数据。
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

# coord = make_coord([h_hr, w_hr], flatten=False)
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):

    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)





