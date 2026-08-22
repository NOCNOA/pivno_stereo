import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from test import eval_stereo
from scheduler import GradualWarmupScheduler
import pandas as pd
import matplotlib.pyplot as plt
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your installation.")
torch.cuda.init()

def sequence_loss(disparity_predictions, disparity_gt):
    """Robust sequence loss over finite positive ground-truth disparity."""
    valid = torch.isfinite(disparity_gt) & (disparity_gt > 0)
    if not valid.any():
        return disparity_predictions[-1].sum() * 0.0

    loss = 0.0
    count = len(disparity_predictions)
    for index, disparity in enumerate(disparity_predictions):
        weight = 0.8 ** (count - index - 1)
        loss = loss + weight * (disparity - disparity_gt).abs()[valid].mean()
    return loss
def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=8, pin_memory=False, persistent_workers=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    #检查配置字典 config 中是否有 resume 字段且它的值是否存在。resume 通常是用于断点续训的文件路径。 暂时没有设置
    if (config.get('resume') is not None) and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'], map_location=torch.device('cpu'))
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        optimizer.param_groups[0]['lr'] = config['optimizer']['args']['lr']
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max'] - config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer, **config['warmup_step_lr'], after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for e in range(1, epoch_start):
            lr_scheduler.step(e)
            # lr_scheduler.step()
        print(epoch_start, optimizer.param_groups[0]['lr'])
    else:
        #如果没有找到可以恢复的检查点文件（即不是断点续训），则根据配置文件初始化新的模型，并将模型加载到 GPU。
        model = models.make(config['model']).cuda()
        #初始化新的优化器，使用模型的参数和配置文件中的优化器设置。
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        # 设置起始训练轮次为 1，这是新模型训练的第一轮。
        epoch_start = 1
        #根据是否存在 multi_step_lr，再次配置学习率调度器，逻辑与上面相同。 暂时不设置
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max'] - config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer, **config['warmup_step_lr'], after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    #计算并记录模型的参数数量，用于日志记录和模型分析。
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, \
          epoch):
    model.train()
    train_loss = utils.Averager()

    pbar = tqdm(train_loader, leave=False, desc='train')
    for batch in pbar:
        img1, img2, disparity_gt = batch
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)
        disparity_gt = disparity_gt.cuda(non_blocking=True)

        predictions = model(img1, img2)
        loss = sequence_loss(predictions, disparity_gt)
        # 计算损失 (假设使用某种损失函数)

        train_loss.add(loss.item())
        # 优化器梯度归零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()
        pbar.set_description('train {:.4f}'.format(train_loss.item()))
        # 更新进度条显示
        #pbar.set_postfix({'loss': loss.item()})


    return train_loss.item()

#   save_path： ./save/_eyes-train_edsr-sronet
def main(config_, save_path):
    # 全局变量
    global config, log, writer
    config = config_
    #   log:  log("Training started.") 将这些信息打印到控制台，并保存到一个日志文件（如 log.txt）。
    #   writer将训练过程中的各种信息（如标量、图像、模型权重等）记录下来，以便在 TensorBoard 中进行可视化的工具
    #   writer有很多种用法   但是这两个都是保存在save_path中   save_path路径存在并以 _ 开头：自动删除并重建 路径存在且不以 _ 开头：询问用户是否删除。
    log, writer = utils.set_save_path(save_path, remove=True)
    #   将配置字典 config 保存到一个 YAML 文件 config.yaml 中，并将该文件存储到指定的路径 save_path 下
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    # 如果可用的 GPU 数量大于 1，则使用 nn.parallel.DataParallel 来启用多 GPU 训练模式
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
    # 配置中的训练参数
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = 1e18
    # 初始化一个计时器 timer，通常用于记录训练过程中的时间。
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        # 记录训练轮次和学习率,记录学习率到 writer，方便后续可视化。
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, \
                           epoch)
        if lr_scheduler is not None:
            # lr_scheduler.step()
            lr_scheduler.step(epoch)
        # 记录训练损失
        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        # 保存模型
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        #定期保存模型
        if ((epoch_save is not None) and (epoch % epoch_save == 0)):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        # 验证模型
        if (epoch == 1) or ((epoch_val is not None) and (epoch % epoch_val == 0)):
            if n_gpus > 1:  # and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_metrics = eval_stereo(val_loader, model_)
            val_res = val_metrics['epe']

            log_info.append('EPE={:.4f}, Bad3={:.2f}%, D1={:.2f}%'.format(
                val_res, 100 * val_metrics['bad3'], 100 * val_metrics['d1']))
            #             writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res < max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        # 记录时间与日志
        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_stereo.yaml')
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    # 限制程序看到的 GPU 数量和具体的 GPU  DataParallel 会自动检测可见的所有 GPU，默认会在所有这些 GPU 上并行运行模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 从 YAML 文件中加载训练或模型的参数配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # split('/') 方法将配置文件路径按 / 分割，获取路径的最后一部分  [:-len('.yaml')]：去掉文件的 .yaml 后缀部分  最终，save_name 会成为 '_train_edsr-sronet'
    save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]

    # os.path.join('./save', '_eyes-train_edsr-sronet')   这行代码会自动在 './save' 和 '_eyes-train_edsr-sronet' 之间添加适当的路径分隔符 /
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
