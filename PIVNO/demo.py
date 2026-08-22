import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import models


IMAGE_EXTENSIONS = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


def collect_stereo_pairs(folder):
    left_files = []
    right_files = []
    for filename in sorted(os.listdir(folder)):
        lower = filename.lower()
        if not lower.endswith(IMAGE_EXTENSIONS):
            continue
        path = os.path.join(folder, filename)
        if 'img1' in lower or 'left' in lower:
            left_files.append(path)
        elif 'img2' in lower or 'right' in lower:
            right_files.append(path)

    if len(left_files) != len(right_files):
        raise ValueError(
            f'left/right image counts differ: {len(left_files)} != {len(right_files)}'
        )
    if not left_files:
        raise ValueError('no stereo pairs found; use left/right or img1/img2 in filenames')
    return zip(left_files, right_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input_images')
    parser.add_argument('--output', default='./output_images')
    parser.add_argument('--model', default='./save/_train_stereo/epoch-best.pth')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    checkpoint = torch.load(args.model, map_location='cpu')
    model = models.make(checkpoint['model'], load_sd=True).cuda().eval()
    os.makedirs(args.output, exist_ok=True)

    to_tensor = transforms.ToTensor()
    for index, (left_path, right_path) in enumerate(collect_stereo_pairs(args.input)):
        left = to_tensor(Image.open(left_path).convert('L')).cuda()
        right = to_tensor(Image.open(right_path).convert('L')).cuda()
        if left.shape != right.shape:
            raise ValueError(f'image size mismatch: {left_path} and {right_path}')

        with torch.no_grad():
            disparity = model(left.unsqueeze(0), right.unsqueeze(0))[-1]

        disparity = disparity[0, 0].cpu().numpy()
        npy_path = os.path.join(args.output, f'disparity_{index:04d}.npy')
        png_path = os.path.join(args.output, f'disparity_{index:04d}.png')
        np.save(npy_path, disparity.astype(np.float32))
        plt.imsave(png_path, disparity, cmap='magma')
        print(f'Saved disparity to {npy_path} and {png_path}')
