import argparse
import os
import torch
import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import GradualWarmupScheduler
from preprocess_for_finetune import ImagePairDataset,apply_displacement,DownsampledFast
from utils import  Averager, make_optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import time
def compute_avg_correlation(img1, img1_pred, kernel_size=9, epsilon=0.01, gamma=0.45):

    padding = kernel_size // 2

    # 提取滑动窗口视图
    unfold_img1 = F.unfold(img1, kernel_size=kernel_size, padding=padding)
    unfold_img1_pred = F.unfold(img1_pred, kernel_size=kernel_size, padding=padding)

    # 计算点积
    dot_product = (unfold_img1 * unfold_img1_pred).sum(dim=1)

    # 计算模值
    norm_img1 = torch.sqrt((unfold_img1**2).sum(dim=1))
    norm_img1_pred = torch.sqrt((unfold_img1_pred**2).sum(dim=1))

    # 计算相关性
    correlation = dot_product / (norm_img1 * norm_img1_pred + 1e-8)  # 避免除以0

    # 所有相关值求平均
    avg_correlation = correlation.mean()
    # 计算 1 - avg_correlation 作为误差 x
    x = 1 - avg_correlation

    # 加入 Charbonnier 惩罚函数
    charbonnier_penalty = (x**2 + epsilon**2)**gamma

    return charbonnier_penalty


def sequence_loss1(img1, img1_pred):
    """Loss function defined over sequence of flow predictions."""

    flow_loss =  (img1_pred[-1] - img1).pow(2).mean()

    return flow_loss

def laplacian_smooth_loss(pred, epsilon=0.000005, gamma=0.45):
    """Laplacian flexible smoothness loss for sequence of flow predictions."""
    loss = 0
    # 定义拉普拉斯核，使用二维卷积核实现
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

    laplacian_kernel = laplacian_kernel.to(pred.device)  # 将核移动到相同的设备

    b, c, h, w = pred.shape
    per_channel_loss = 0
    for j in range(c):
        channel = pred[:, j:j + 1, :, :]  # 取第 j 个通道，形状为 (b, 1, h, w)

        # 计算拉普拉斯变换（卷积运算）
        laplacian_output = F.conv2d(channel, laplacian_kernel, padding=1)  # 保持尺寸不变

        # 在 diff 中移除边界点
        diff = laplacian_output[:, :, 1:-1, 1:-1].pow(2)
        per_channel_loss += diff.sum() / (b * (h - 2) * (w - 2))  # 更新为内部区域

    # 加入 Charbonnier 惩罚函数
    charbonnier_penalty = (per_channel_loss**2 + epsilon**2)**gamma

    return charbonnier_penalty


def gaussian_smooth_loss(pred):
    """Laplacian flexible smoothness loss for sequence of flow predictions."""
    # 定义拉普拉斯核，使用二维卷积核实现
    kernel = torch.tensor([[0, 0, 1, 0, 0],
                           [0, 2, 2, 2, 0],
                           [1, 2, -20, 2, 1],
                           [0, 2, 2, 2, 0],
                           [0, 0, 1, 0, 0]], dtype=torch.float32)

    kernel = kernel.view(1, 1, 5, 5)  # 转换为 (1, 1, 5, 5) 的形状

    gaussian_kernel = kernel.to(pred.device)  # 将核移动到相同的设备

    b, c, h, w = pred.shape

    per_channel_loss = 0
    for j in range(c):
        channel = pred[:, j:j + 1, :, :]  # 取第 j 个通道，形状为 (b, 1, h, w)

        # 计算高斯变换（卷积运算）
        gaussian_output = F.conv2d(channel, gaussian_kernel, padding=2)  # 保持尺寸不变

        # 在 diff 中移除边界点
        diff = gaussian_output[:, :, 2:-2, 2:-2].pow(2)
        per_channel_loss += diff.sum() / (b * (h - 4) * (w - 4))  # 更新为内部区域，考虑到padding


    return per_channel_loss

def divergence(u_x, u_y):
    """Compute the divergence of a 2D vector field (u_x, u_y) with boundaries removed."""
    # 计算离散化的偏导数，去掉边界
    dx = u_x[..., 1:] - u_x[..., :-1]  # x方向偏导数
    dy = u_y[..., 1:, :] - u_y[..., :-1, :]  # y方向偏导数

    # 去掉边界后计算散度
    div_u = dx[..., :-1, :] + dy[..., :, :-1]
    return div_u


def sequence_loss3(pred, epsilon=0.0003, gamma=0.45):
    """Loss function that computes the divergence of flow predictions as loss."""

    # 预测流场分量 u_x 和 u_y
    u_x, u_y = pred[:, 0, :, :], pred[:, 1, :, :]

    # 计算散度
    div_u = divergence(u_x, u_y)

    # 计算散度的平均值作为损失
    div_loss = div_u.abs().mean()

    # 加入 Charbonnier 惩罚函数
    charbonnier_penalty = (div_loss**2 + epsilon**2)**gamma

    return charbonnier_penalty


def prepare_training(config, model_path, device):
    """Prepare model, optimizer, and scheduler for training."""
    # Load model weights
    sv_file = torch.load(model_path, map_location=device)
    model = models.make(sv_file['model'], load_sd=True).to(device)

    # Initialize optimizer
    optimizer = make_optimizer(model.parameters(), config['optimizer'])

    # Configure learning rate scheduler (Remove warmup phase)
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=config['epoch_max']
    )

    print(f"Model loaded for fine-tuning. Initial learning rate: {optimizer.param_groups[0]['lr']}")
    return model, optimizer, lr_scheduler

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train the model for one epoch."""
    model.train()
    train_loss = Averager()
    pbar = tqdm(data_loader, leave=False, desc='train')
    for batch in pbar:
        img1, img2, img1_down, img2_down, coord, cell = batch
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)
        img1_down = img1_down.cuda(non_blocking=True)
        img2_down = img2_down.cuda(non_blocking=True)
        coord = coord.cuda(non_blocking=True)
        cell = cell.cuda(non_blocking=True)
        # 预测
        pred = model(img1_down, img2_down, coord,  cell)[-1]

        img1_pred=apply_displacement(img2, pred)
        # Compute loss
        loss1 = compute_avg_correlation( img1, img1_pred)
        loss2 = laplacian_smooth_loss(pred)*9598
        loss3 = sequence_loss3(pred)*27.6

        loss=loss1+loss2+loss3
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update loss tracker
        train_loss.add(loss.item())
        # pbar.set_description('train {:.4f}'.format(train_loss.item()))
        pbar.set_description(
            'train {:.8f}, loss1 {:.8f}, loss2 {:.8f}, loss3 {:.8f}'.format(
                train_loss.item(), loss1.item(), loss2.item(), loss3.item()
            )
        )
    return train_loss.item()


def fine_tune(args,data_loader):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurations
    config = {
        'epoch_max': args.epochs,
         'optimizer': {'name': 'adam','args': {'lr': 5e-4}
    }
    }

    # Prepare training components
    model, optimizer, lr_scheduler = prepare_training(config, args.model, device)

    best_loss = float('inf')  # 记录最好的损失（可以根据实际需求修改）

    start_time = time.time()  # 记录训练开始的时间
    # Training loop
    for epoch in range(1, config['epoch_max'] + 1):

        epoch_start_time = time.time()  # 记录当前 epoch 开始的时间

        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()

        epoch_end_time = time.time()  # 记录当前 epoch 结束的时间
        epoch_duration = epoch_end_time - epoch_start_time  # 当前 epoch 持续的时间

        print(f"Epoch {epoch}/{config['epoch_max']}, Loss: {train_loss:.4f}, Epoch Time: {epoch_duration:.2f}s")

        # 每 100 个 epoch 保存一次模型
        if epoch % 10 == 0:
            # 保存当前模型的权重
            sv_file = torch.load(args.model, map_location=device)
            model_spec = sv_file['model']  # 完整的 model_spec

            # 更新 model_spec 中的权重
            model_spec['sd'] = model.state_dict()

            # 保存文件时保留完整配置
            save_path = os.path.join(args.output, f"tune-epoch-{epoch}.pth")
            torch.save({
                'model': model_spec,  # 保存完整的 model_spec，包括权重
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, save_path)

        # 更新最好的模型（例如基于最低损失保存最好的模型）
        if train_loss < best_loss:
            best_loss = train_loss
            # 保存最佳模型的权重
            sv_file = torch.load(args.model, map_location=device)
            model_spec = sv_file['model']  # 完整的 model_spec

            # 更新 model_spec 中的权重
            model_spec['sd'] = model.state_dict()

            # 保存文件时保留完整配置
            save_path = os.path.join(args.output, "tune-best.pth")
            torch.save({
                'model': model_spec,  # 保存完整的 model_spec，包括权重
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, save_path)
    total_time = time.time() - start_time  # 训练总时间
    print(f"Training completed. Total time: {total_time:.2f}s")
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input_images')
    parser.add_argument('--output', default='./output_images')
    parser.add_argument('--model', default='./save/_train_PIV-genImages/epoch-best.pth', help="Path to pre-trained model")
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for fine-tuning')
    arg = parser.parse_args()

    # 获取文件夹路径
    input_folder = arg.input
    # 创建 output 文件夹
    os.makedirs(arg.output, exist_ok=True)

    dataset = ImagePairDataset(input_folder,phase=1)
    dataset = DownsampledFast(dataset=dataset, phase=1)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    #微调训练
    fine_tune(arg, data_loader)
