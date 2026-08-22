import argparse
import os
import numpy as np
import torch
import cv2
import models
from preprocess_for_finetune import ImagePairDataset,DownsampledFast,apply_displacement
from torch.utils.data import DataLoader
from tqdm import tqdm
from fine_tune_train import sequence_loss1,laplacian_smooth_loss,sequence_loss3
import matplotlib.pyplot as plt
import numpy as np
import math
from fine_tune_train import compute_avg_correlation
def load_flow( file_path):
    """读取 .flo 文件的函数"""
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f'Magic number incorrect in file {file_path}')

        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        data = np.frombuffer(f.read(), dtype=np.float32)
        U = data[:width * height].reshape((height, width))
        V = data[width * height:].reshape((height, width))
        flow = np.stack((U, V), axis=-1)
    print(f"Loaded flow shape: {flow.shape}, dtype: {flow.dtype}, range: ({flow.min()}, {flow.max()})")
    return flow

def compute_optical_flow_divergence(flow):
    """
    计算实测光流场的平均散度 (Average Divergence)

    参数:
        flow: ndarray, 实测流场数据，形状为 (H, W, 2)

    返回:
        avg_divergence: 平均散度
    """
    # 获取光流分量
    u = flow[..., 0]  # x方向分量
    v = flow[..., 1]  # y方向分量

    # 计算偏导数
    du_dx = np.gradient(u, axis=1)  # u 对 x 的偏导数
    dv_dy = np.gradient(v, axis=0)  # v 对 y 的偏导数

    # 计算散度
    divergence = du_dx + dv_dy

    # 计算平均散度
    sum_divergence = np.sum(divergence)

    return sum_divergence

# print('aee=',compute_optical_flow_aee(flow))
def compute_optical_flow_aee(flow):
    """
    计算实测光流场与理论光流场之间的平均端点误差 (AEE)

    参数:
        flow: ndarray, 实测流场数据，形状为 (H, W, 2)

    返回:
        aee: 平均端点误差 (Average Endpoint Error)
    """
    # 计算模值
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # 找到最小值的坐标
    min_index = np.unravel_index(np.argmin(magnitude), magnitude.shape)
    print(min_index)
    # 设置参数
    omega = 10 * np.pi / 9      # 恒定角速度，rad/s
    delta_t = 4e-3              # 时间间隔，4 ms
    x_c, y_c = 543, 508 # 旋转中心为图像中心
    # x_c, y_c = min_index[1], min_index[0] # 旋转中心为图像中心
    # x_c, y_c = flow.shape[1] // 2, flow.shape[0] // 2  # 旋转中心为图像中心
    # 生成网格坐标
    x = np.arange(flow.shape[1])  # 水平坐标
    y = np.arange(flow.shape[0])  # 垂直坐标
    xx, yy = np.meshgrid(x, y)

    # 理论光流计算
    u_theory = -omega * delta_t * (yy - y_c)  # x方向理论位移
    v_theory = omega * delta_t * (xx - x_c)   # y方向理论位移
    # 实测光流分量
    u_measured = flow[..., 0]
    v_measured = flow[..., 1]
    theory_flow = np.stack((u_theory, v_theory), axis=-1)
    save_flow_with_magnitude_background(theory_flow, "GT.png")
    # 计算AEE
    error = np.sqrt((u_measured - u_theory) ** 2 + (v_measured - v_theory) ** 2)
    aee = np.mean(error)

    return aee

# #save_flow_visualization_row(flow,img1[i], output_path)
def save_flow_visualization_row(flow, img, output_path, scale=8):
    """保存光流可视化图像_原图像作为背景"""
    height, width, _ = flow.shape
    xx, yy = np.meshgrid(range(width), range(height))
    img_resized = cv2.resize(img.squeeze(0).cpu().numpy(), (width, height))
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # 映射颜色到位移长度
    norm = plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max())
    colors = plt.cm.viridis(norm(magnitude[::scale, ::scale])).reshape(-1, 4)

    # 绘制光流图
    plt.figure(figsize=(20, 20))
    plt.imshow(img_resized, cmap='gray')
    plt.quiver(
        xx[::scale, ::scale], yy[::scale, ::scale],
        flow[..., 0][::scale, ::scale],
        -flow[..., 1][::scale, ::scale],
        color=colors, #scale=1, scale_units='xy'
    )
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved flow visualization to {output_path}")

def save_flow_with_magnitude_background(flow, output_path, scale=20, colormap='jet'):

    height, width, _ = flow.shape
    xx, yy = np.meshgrid(range(width), range(height))

    # 计算光流矢量的幅值
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # 对光流进行归一化到0-1之间
    normalized_flow = flow / (np.expand_dims(magnitude, axis=-1) + 1e-8)
    normalized_magnitude = magnitude / (np.max(magnitude) + 1e-8)  # 将幅值归一化到0-1
    adjusted_magnitude = np.maximum(normalized_magnitude, 0.3)  # 幅值小于0.3的调整为0.3
    adjusted_flow = normalized_flow * np.expand_dims(adjusted_magnitude, axis=-1)*np.max(magnitude)

    # 设置速度幅值范围
    vmax = math.ceil(np.percentile(magnitude, 99))  # vmax 向上取整
    vmax =10
    vmin = 0  # vmin 固定为 0

    # 映射背景颜色到实际速度幅值范围
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)

    # 绘制背景（光流幅值图）
    plt.figure(figsize=(20, 20))
    im = plt.imshow(magnitude, cmap=cmap, norm=norm, origin='upper', extent=(0, width, height, 0))

    # 添加颜色条，并使其紧挨图片且缩短
    cbar = plt.colorbar(im, fraction=0.046, pad=0.01, shrink=0.75)  # 调整位置和长度
    cbar.ax.tick_params(labelsize=30)
    # 设置颜色条刻度
    num_ticks = 6
    tick_values = np.linspace(vmin, vmax, num=num_ticks)
    cbar.set_ticks(tick_values)
    cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in tick_values])  # 刻度保留一位小数

    # 绘制光流矢量
    plt.quiver(
        xx[::scale, ::scale], yy[::scale, ::scale],
        normalized_flow[..., 0][::scale, ::scale],
        normalized_flow[..., 1][::scale, ::scale],
        color='black', angles='xy', scale_units='xy', scale=0.1, pivot='middle'
    )

    # 去掉坐标轴和边框
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()
    print(f"Saved flow visualization with magnitude background to {output_path}")
    """保存光流图像，背景颜色表示速度幅值，并在右侧显示颜色对应的位移量"""


# save_flow_with_streamlines_background(flow,  output_path)
def save_flow_with_streamlines_background(flow, output_path, colormap='jet', density=0.7):
    """保存光流图像，背景颜色表示速度幅值，并使用流线替代光流矢量"""
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    height, width, _ = flow.shape
    xx, yy = np.meshgrid(range(width), range(height))

    # 计算光流矢量的幅值和方向
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # 设置速度幅值范围
    vmax = math.ceil(np.percentile(magnitude, 99))  # vmax 向上取整
    vmax=7
    vmin = 0  # vmin 固定为 0

    # 映射背景颜色到实际速度幅值范围
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)

    # 绘制背景（光流幅值图）
    plt.figure(figsize=(20, 20))
    plt.imshow(magnitude, cmap='viridis', norm=norm, origin='upper', extent=(0, width, height, 0))

    # 添加颜色条
    cbar = plt.colorbar(fraction=0.046, pad=0.01, shrink=1)  # 调整位置和长度
    cbar.ax.tick_params(labelsize=30)
    tick_values = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_values)
    cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in tick_values])

    # 绘制流线图
    plt.streamplot(
        xx, yy, flow[..., 0], flow[..., 1],
        color='w', linewidth=1.2, density=density, arrowsize=2
    )

    # 去掉坐标轴和边框
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()
    print(f"Saved flow visualization with streamlines background to {output_path}")

# plot_velocity_histograms(preds,  args.output)
def plot_velocity_histograms(preds, output_dir):
    """
    绘制 x 方向、y 方向和速度模值的直方图，并保存到指定目录。

    """
    # 绘制 x 方向速度分量直方图
    # 提取 x 和 y 方向的速度分量
    x_velocity = preds[0, 0, :, :].cpu().numpy().flatten()
    y_velocity = preds[0, 1, :, :].cpu().numpy().flatten()
    # 计算模值
    total_magnitude = np.sqrt(x_velocity ** 2 + y_velocity ** 2)
    plt.figure(figsize=(10, 5))
    plt.hist(x_velocity, bins=1000, color='blue', alpha=0.7)
    plt.title('Histogram of X Velocity')
    plt.xlabel('X Velocity')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(int(x_velocity.min()), int(x_velocity.max()) + 1, 1))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/x_velocity_histogram.png')
    plt.close()

    # 绘制 y 方向速度分量直方图
    plt.figure(figsize=(10, 5))
    plt.hist(y_velocity, bins=1000, color='green', alpha=0.7)
    plt.title('Histogram of Y Velocity')
    plt.xlabel('Y Velocity')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(int(y_velocity.min()), int(y_velocity.max()) + 1, 1))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/y_velocity_histogram.png')
    plt.close()

    # 绘制速度模值直方图
    plt.figure(figsize=(10, 5))
    plt.hist(total_magnitude, bins=1000, color='red', alpha=0.7)
    plt.title('Histogram of Velocity Magnitude')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(int(total_magnitude.min()), int(total_magnitude.max()) + 1, 1))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/magnitude_histogram.png')
    plt.close()

def compute_optical_flow_aee_caseA(flow,batch_idx):
    print(flow.shape)
    piv_x = np.arange(24, 984+8, 8)  # X 采样点
    piv_y = np.arange(32, 992+8, 8)  # Y 采样点

    sampled_flow = flow[np.ix_(piv_y, piv_x)]  # 结果为 (121, 121, 2)

    file_path = f'jetdata_piv_flow/frame_{batch_idx+1:03d}.flo'
    flow_data = load_flow(file_path) # 输出应为 (height, width, 2)
    print(flow_data.shape)
    save_flow_with_streamlines_background(flow_data,'flow.png')
    save_flow_with_streamlines_background(flow,'pred_flow.png')
    error = np.sqrt(np.sum((sampled_flow - flow_data) ** 2, axis=-1))
    aee = np.mean(error)
    print(aee)
    return aee

def compute_optical_flow_aee_SPID(flow,batch_idx):
    file_path = f'./SPID/dataset/SPID_{batch_idx+1:04d}.flo'
    # file_path = './SPID/dataset/SPID_0019.flo'
    flow_data = read_flo_file(file_path) # 输出应为 (height, width, 2)
    # save_flow_with_streamlines_background(flow_data,f'./SPID/flow{batch_idx+1:04d}.png')
    # save_flow_with_magnitude_background(flow,f'./SPID/pred_flow{batch_idx+1:04d}.png')
    # save_flow_with_magnitude_background(flow,f'./pred_flow{batch_idx+1:04d}.png')
    error = np.sqrt(np.sum((flow - flow_data) ** 2, axis=-1))
    aee = np.mean(error)
    print(aee)
    return aee

def compute_optical_flow_aee_case1(flow,batch_idx):
    # file_path = f'./SPID/dataset/SPID_{batch_idx+1:04d}.flo'
    file_path = F'./DNS/DNS_turbulence_{batch_idx+1:05d}_flow.flo'
    flow_data = load_flow(file_path) # 输出应为 (height, width, 2)
    # save_flow_with_magnitude_background(flow_data,f'./flow{batch_idx+1:04d}.png')
    # save_flow_with_magnitude_background(flow,f'./SPID/pred_flow{batch_idx+1:04d}.png')
    # save_flow_with_magnitude_background(flow,f'./pred_flow{batch_idx+1:04d}.png')
    error = np.sqrt(np.sum((flow - flow_data) ** 2, axis=-1))
    aee = np.mean(error)
    print(aee)
    return aee

def read_flo_file(file_path):
    """
    Read an optical flow file in .flo format.

    Parameters:
        file_path (str): Path to the .flo file.

    Returns:
        ndarray: Optical flow array with shape (2, H, W).
    """
    with open(file_path, 'rb') as f:
        # Check magic number
        magic = f.read(4)
        if magic != b'PIEH':
            raise ValueError(f"Invalid .flo file: incorrect magic number in {file_path}.")

        # Read dimensions
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Read flow data
        flow_data = np.frombuffer(f.read(), dtype=np.float32)
        flow = flow_data.reshape((height, width, 2)) # (H, W, 2)
    return flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input_images')
    parser.add_argument('--output', default='./output_images')
    parser.add_argument('--model', default='./output_images/tune-best.pth', help="Path to pre-trained model")
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.output, exist_ok=True)

    # 数据集加载
    dataset = DownsampledFast(dataset=ImagePairDataset(args.input, phase=0), phase=0)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 模型加载
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    aee=[]
    divergence=[]

    # 推理与可视化
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, leave=False, desc='Testing')):
            img1, img2, img1_down, img2_down, coord, cell = (x.cuda(non_blocking=True) for x in batch)
            pred = model(img1_down, img2_down, coord, cell)
            preds = pred[-1]
            # img1_pred = apply_displacement(img2, preds)
            # Compute loss
            # loss1 = compute_avg_correlation(img1, img1_pred)
            # print(loss1)
            for i in range(preds.shape[0]):
                flow = preds[i].permute(1, 2, 0).cpu().numpy()
                output_path = os.path.join(args.output, f"output_flow_{batch_idx:04d}_{i:02d}.png")
                # 4f
                # save_flow_with_magnitude_background(flow,  output_path)
                # plot_velocity_histograms(preds, args.output)
                # print('aee=',compute_optical_flow_aee(flow))
                # print('divergence=',compute_optical_flow_divergence(flow))
                # 2a
                aee.append(compute_optical_flow_aee_caseA(flow,batch_idx))
                # aee.append(compute_optical_flow_aee_case1(flow, batch_idx))
                # aee.append(compute_optical_flow_aee_SPID(flow, batch_idx))
                # divergence.append(compute_optical_flow_divergence(flow))
            del img1, img2, img1_down, img2_down, preds, coord, cell
            torch.cuda.empty_cache()
    average_aee = np.mean(aee)
    average_divergence=np.mean(divergence)
    print(f"Average AEE over all batches: {average_aee:.4f}")
    print(f"Average divergence over all batches: {average_divergence:.4f}")

if __name__ == '__main__':
    main()