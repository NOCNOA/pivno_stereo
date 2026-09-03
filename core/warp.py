import torch
import torch.nn.functional as F


def meshgrid(img):
    """
    img: [B, C, H, W]
    return: [B, 2, H, W]
    """
    b, _, h, w = img.shape
    device = img.device
    dtype = img.dtype

    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, w).expand(1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, h, 1).expand(1, h, w)

    grid = torch.cat((x, y), dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, 2, H, W]

    return grid


def normalize_coords(grid, align_corners=True):
    """
    grid: [B, 2, H, W]
    return: [B, H, W, 2]
    """
    b, _, h, w = grid.shape

    x = grid[:, 0]
    y = grid[:, 1]

    if align_corners:
        x = 2 * x / (w - 1) - 1
        y = 2 * y / (h - 1) - 1
    else:
        x = 2 * (x + 0.5) / w - 1
        y = 2 * (y + 0.5) / h - 1

    return torch.stack((x, y), dim=-1)


def interp(x, sample_grid, padding_mode='border', align_corners=True):
    original_dtype = x.dtype

    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.grid_sample(
            x.float(),
            sample_grid.float(),
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=align_corners
        )

    return output_fp32.to(original_dtype)


def disp_warp(img, disp, padding_mode='border', align_corners=True):
    """
    Warp right image/feature to left view using left disparity.

    Args:
        img:  [B, C, H, W], usually right image/feature
        disp: [B, 1, H, W], positive left disparity
    Returns:
        warped_img: [B, C, H, W]
    """
    grid = meshgrid(img)  # [B, 2, H, W]

    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = grid + offset  # [B, 2, H, W]

    sample_grid_norm = normalize_coords(sample_grid, align_corners=align_corners)

    warped_img = interp(
        img,
        sample_grid_norm,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    return warped_img


def disp_warp_multi(img, disp_samples, padding_mode='border', align_corners=True):
    """
    Warp right feature with multiple disparity hypotheses.

    Args:
        img: [B, C, H, W]
        disp_samples: [B, S, H, W] or [B, S, 1, H, W]
    Returns:
        warped: [B, S, C, H, W]
    """
    if disp_samples.ndim == 5:
        disp_samples = disp_samples.squeeze(2)
    if disp_samples.ndim != 4:
        raise ValueError(f"disp_samples must have shape [B, S, H, W] or [B, S, 1, H, W], got {disp_samples.shape}")

    b, s, h, w = disp_samples.shape
    c = img.shape[1]

    img_rep = img.unsqueeze(1).expand(b, s, c, h, w).reshape(b * s, c, h, w)
    disp_rep = disp_samples.reshape(b * s, 1, h, w)
    warped = disp_warp(img_rep, disp_rep, padding_mode=padding_mode, align_corners=align_corners)

    return warped.view(b, s, c, h, w)
