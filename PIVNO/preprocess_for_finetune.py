import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import torch.nn.functional as F
from utils import make_coord
from torchvision.transforms import InterpolationMode
import os
import math

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))
class ImagePairDataset(Dataset):
    def __init__(self, input_folder, phase):
        self.phase = phase
        # 定义图像转换
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 初始化两个列表，分别存储 img1 和 img2 开头的图片
        self.img1_files = []
        self.img2_files = []

        # 遍历文件夹中的所有文件
        for filename in sorted(os.listdir(input_folder)):
            filepath = os.path.join(input_folder, filename)

            # 检查文件名中是否包含 'img1' 或 'img2'，并且是图片文件
            if 'img1' in filename and filename.endswith(('.png', '.bmp', '.jpg', '.jpeg', '.tif')):
                self.img1_files.append(filepath)
            elif 'img2' in filename and filename.endswith(('.png', '.bmp', '.jpg', '.jpeg', '.tif')):
                self.img2_files.append(filepath)

        # 确保 img1 和 img2 数量一致
        assert len(self.img1_files) == len(self.img2_files), "img1 和 img2 数量不匹配"


    def __len__(self):
        if self.phase == 1:
            return len(self.img1_files)*16
        else: return len(self.img1_files)

    def __getitem__(self, idx):
        img1 = self.load_image(self.img1_files[idx%len(self.img1_files)])
        img2 = self.load_image(self.img2_files[idx%len(self.img1_files)])
        # 图像转换为张量并保存到内存
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        return img1, img2

    @staticmethod
    def load_image(filepath):
        from PIL import Image
        image = Image.open(filepath).convert('L')
        return image


def apply_displacement(img, displacement_field):
    """
    根据位移场生成第二帧图像
    """
    b, c, h, w = img.shape
    assert c == 1, "图像的通道数必须为 1"

    assert displacement_field.shape == (b, 2, h, w), \
        "位移场形状必须为 (b, 2, h, w)"
    # 创建归一化坐标网格
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device),
        torch.linspace(-1, 1, w, device=img.device),
        indexing="ij"
    )  # 归一化范围为 [-1, 1]
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)  # (b, h, w, 2)

    displacement_field[:, 0, :, :] /= (w/2)  # 水平方向归一化
    displacement_field[:, 1, :, :] /= (h/2)  # 垂直方向归一化

    # 计算新的坐标

    device = torch.device("cuda:0")
    grid = grid.to(device)
    displacement_field = displacement_field.to(device)

    uv = grid + displacement_field.permute(0, 2, 3, 1)  # 位移场 shape (b, h, w, 2)

    # 使用 grid_sample 插值
    img1 = F.grid_sample(img, uv, mode='bicubic', padding_mode='reflection', align_corners=True)  # (b, 1, h, w)

    return img1


class DownsampledFast(Dataset):

    def __init__(self, dataset, phase=2 ):
        self.dataset = dataset
        self.phase = phase
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, img2 = self.dataset[idx]
        #获取 gt
        h = img1.shape[-2]
        w = img1.shape[-1]
        scale = max(h, w) / 256
        scale=4
        if self.phase == 1:     ##微调阶段
            h_lr = math.floor(img1.shape[-2] / scale/2 + 1e-9)
            w_lr = math.floor(img1.shape[-1] / scale/2 + 1e-9)
            h_hr = round(h_lr * scale)
            w_hr = round(w_lr * scale)
            x0 = random.randint(0, img1.shape[-2] - h_hr)
            y0 = random.randint(0, img1.shape[-1] - w_hr)
            hr_img1 = img1[:, x0: x0 + h_hr, y0: y0 + w_hr]
            hr_img2 = img2[:, x0: x0 + h_hr, y0: y0 + w_hr]
        else:
            h_lr = math.floor(img1.shape[-2] / scale + 1e-9)
            w_lr = math.floor(img1.shape[-1] / scale + 1e-9)
            h_hr = img1.shape[-2]
            w_hr = img1.shape[-1]
            hr_img1 = img1
            hr_img2 = img2

        img1_down = resize_fn(hr_img1, (h_lr, w_lr))
        img2_down = resize_fn(hr_img2, (h_lr, w_lr))

        coord = make_coord([h_hr, w_hr], flatten=False)

        cell = torch.ones(2)
        cell[0] *= 2 / h_hr
        cell[1] *= 2 / w_hr

        return hr_img1, hr_img2, img1_down, img2_down, coord, cell


# if __name__ == "__main__":


