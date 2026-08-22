import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models


def eval_stereo(loader, model):
    """Return pixel-weighted EPE, Bad-3 and KITTI-style D1 metrics."""
    model.eval()
    total_error = 0.0
    total_bad3 = 0
    total_d1 = 0
    total_valid = 0

    with torch.no_grad():
        for left, right, disparity_gt in tqdm(loader, leave=False, desc='val'):
            left = left.cuda(non_blocking=True)
            right = right.cuda(non_blocking=True)
            disparity_gt = disparity_gt.cuda(non_blocking=True)

            disparity = model(left, right)[-1]
            valid = torch.isfinite(disparity_gt) & (disparity_gt > 0)
            error = (disparity - disparity_gt).abs()
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                continue

            valid_error = error[valid]
            valid_gt = disparity_gt[valid]
            total_error += valid_error.sum().item()
            total_bad3 += (valid_error > 3.0).sum().item()
            total_d1 += (
                (valid_error > 3.0) & (valid_error / valid_gt > 0.05)
            ).sum().item()
            total_valid += valid_count

    if total_valid == 0:
        raise RuntimeError('validation set contains no finite positive disparities')
    return {
        'epe': total_error / total_valid,
        'bad3': total_bad3 / total_valid,
        'd1': total_d1 / total_valid,
    }


def eval_psnr(loader, model):
    """Backward-compatible name used by older training scripts."""
    return eval_stereo(loader, model)['epe']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_stereo.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )

    checkpoint = torch.load(config['model_path'], map_location='cpu')
    model = models.make(checkpoint['model'], load_sd=True).cuda()
    metrics = eval_stereo(loader, model)

    test_type = spec['dataset']['args'].get('type', 'all')
    print(
        f'Stereo results: {test_type} '
        f'EPE={metrics["epe"]:.4f}, '
        f'Bad3={100 * metrics["bad3"]:.2f}%, '
        f'D1={100 * metrics["d1"]:.2f}%'
    )
