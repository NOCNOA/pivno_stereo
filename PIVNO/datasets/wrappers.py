import random

from torch.utils.data import Dataset

from datasets import register


@register('sr-implicit-downsampled-fast')
class StereoCrop(Dataset):
    """Keep stereo images at native resolution and optionally crop them."""

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 batch_size=10, random_crop=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.random_crop = random_crop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        left, right, disparity = self.dataset[idx]
        if left.shape[-2:] != right.shape[-2:]:
            raise ValueError('left and right images must have identical sizes')
        if left.shape[-2:] != disparity.shape[-2:]:
            raise ValueError('image and disparity sizes must match')

        if self.inp_size is None:
            return left, right, disparity

        if isinstance(self.inp_size, int):
            crop_h = crop_w = self.inp_size
        elif isinstance(self.inp_size, (list, tuple)) and len(self.inp_size) == 2:
            crop_h, crop_w = map(int, self.inp_size)
        else:
            raise ValueError('inp_size must be an int or [height, width]')
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError('crop dimensions must be positive')
        height, width = left.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError(
                f'crop size {self.inp_size} exceeds sample size {(height, width)}'
            )

        if self.random_crop:
            top = random.randint(0, height - crop_h)
            x0 = random.randint(0, width - crop_w)
        else:
            top = (height - crop_h) // 2
            x0 = (width - crop_w) // 2

        region = (..., slice(top, top + crop_h), slice(x0, x0 + crop_w))
        return left[region], right[region], disparity[region]
