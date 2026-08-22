from pathlib import Path

import numpy as np
import torch
from PIL import Image


def resolve_list_path(path, list_path):
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return Path(list_path).resolve().parent / path


def read_pfm(path):
    with open(path, 'rb') as stream:
        header = stream.readline().decode('ascii').rstrip()
        if header not in ('PF', 'Pf'):
            raise ValueError(f'invalid PFM header in {path}')
        width, height = map(int, stream.readline().decode('ascii').split())
        scale = float(stream.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        channels = 3 if header == 'PF' else 1
        data = np.fromfile(stream, endian + 'f')
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f'invalid PFM payload in {path}: {data.size} != {expected}')
    data = np.flipud(data.reshape(height, width, channels))
    return data[..., 0]


def read_flo_horizontal(path):
    with open(path, 'rb') as stream:
        magic = np.fromfile(stream, np.float32, count=1)
        if magic.size != 1 or magic[0] != 202021.25:
            raise ValueError(f'invalid .flo header in {path}')
        width = int(np.fromfile(stream, np.int32, count=1)[0])
        height = int(np.fromfile(stream, np.int32, count=1)[0])
        flow = np.fromfile(stream, np.float32, count=2 * width * height)
    return flow.reshape(height, width, 2)[..., 0]


def load_disparity(path, disparity_scale=1.0):
    """Load disparity as a contiguous float tensor with shape [1, H, W]."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        disparity = np.load(path)
    elif suffix == '.pfm':
        disparity = read_pfm(path)
    elif suffix == '.flo':
        # Useful for legacy lists: only horizontal displacement is retained.
        disparity = read_flo_horizontal(path)
    else:
        disparity = np.asarray(Image.open(path))

    disparity = np.asarray(disparity)
    if disparity.ndim == 3:
        if disparity.shape[0] == 1:
            disparity = disparity[0]
        else:
            disparity = disparity[..., 0]
    if disparity.ndim != 2:
        raise ValueError(f'disparity must be 2-D, got {disparity.shape} from {path}')
    if disparity_scale <= 0:
        raise ValueError('disparity_scale must be positive')

    disparity = np.ascontiguousarray(disparity, dtype=np.float32)
    disparity /= float(disparity_scale)
    return torch.from_numpy(disparity).unsqueeze(0)
