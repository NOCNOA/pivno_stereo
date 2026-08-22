from __future__ import print_function, division
import sys

import argparse
import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.defom_pact import DEFOMStereo as PACTDEFOMStereo
from core.defom_pact_smd import DEFOMStereo as PACTSMDDEFOMStereo
from core.defom_pact_smd_post import DEFOMStereo as PACTSMDPostDEFOMStereo
from core.defom_pact_bilap_gru import DEFOMStereo as PACTBiLapGRUDEFOMStereo
from core.defom_pact2 import DEFOMStereo as PACT2DEFOMStereo
from core.defom_pact2_gev import DEFOMStereo as PACT2GEVDEFOMStereo
from core.pivno_models.defom_pact_pivno import DEFOMStereo as PACTPIVNODEFOMStereo
from core.pivno_models.defom_pivno import DEFOMStereo as PIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated import DEFOMStereo as GatedPIVNODEFOMStereo
try:
    from core.defom_cor_ga import DEFOMStereo as CorGADEFOMStereo, autocast
except ModuleNotFoundError:
    CorGADEFOMStereo = None
    autocast = torch.cuda.amp.autocast

import core.stereo_datasets as datasets
from core.utils.utils import InputPadder

import os
os.environ.pop("CUDNN_DISABLE", None)  # 防止环境变量禁用
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

if int(os.environ.get("RANK", "0")) == 0:
    print("cudnn.enabled =", torch.backends.cudnn.enabled)
    print("cudnn.deterministic =", torch.backends.cudnn.deterministic)
    print("cudnn.benchmark =", torch.backends.cudnn.benchmark)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def _checkpoint_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"checkpoint must be a mapping, got {type(checkpoint).__name__}"
        )
    state = checkpoint.get('model', checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint['model'] must be a state-dict mapping")
    if state and all(key.startswith('module.') for key in state):
        state = {
            key[len('module.'):]: value
            for key, value in state.items()
        }
    return state


def _infer_pivno_input_channels(checkpoint):
    config = checkpoint.get('model_config', {})
    if isinstance(config, dict) and 'pivno_input_channels' in config:
        channels = int(config['pivno_input_channels'])
    else:
        state = _checkpoint_model_state(checkpoint)
        key = 'pivno.snet.conv1.weight'
        if key not in state:
            raise KeyError(f"PIVNO checkpoint is missing {key!r}")
        weight = state[key]
        if weight.ndim != 4:
            raise ValueError(
                f"{key} must be a Conv2d weight, got {tuple(weight.shape)}"
            )
        channels = int(weight.shape[1])
    if channels not in (1, 3):
        raise ValueError(f"unsupported PIVNO input channel count: {channels}")
    return channels

def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _distributed_enabled() else 0


def _world_size():
    return dist.get_world_size() if _distributed_enabled() else 1


def _is_main_process():
    return _rank() == 0


def _validation_indices(dataset_size):
    return range(_rank(), dataset_size, _world_size())


def _sum_across_processes(values):
    tensor = torch.tensor(
        values,
        dtype=torch.float64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if _distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def _require_nonempty_evaluation(sample_count, dataset_name):
    if sample_count <= 0:
        raise ValueError(
            f"{dataset_name} evaluation found no samples; check the dataset root"
        )


def _distributed_nanmean_rows(rows):
    values = np.asarray(rows, dtype=np.float64)
    local_sums = np.nansum(values, axis=1)
    local_counts = np.isfinite(values).sum(axis=1)
    reduced = _sum_across_processes(
        np.concatenate((local_sums, local_counts)).tolist()
    )
    row_count = values.shape[0]
    sums = reduced[:row_count]
    counts = reduced[row_count:]
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )


def _model_manages_amp(model):
    return isinstance(
        _unwrap_model(model),
        (PACTDEFOMStereo, PACTSMDDEFOMStereo, PACT2DEFOMStereo),
    )


def _eval_forward(model, mixed_prec, *args, **kwargs):
    """PACT owns its AMP boundaries; legacy models retain evaluator autocast."""
    if _model_manages_amp(model):
        output = model(*args, **kwargs)
    else:
        with autocast(enabled=mixed_prec):
            output = model(*args, **kwargs)
    if isinstance(output, dict) and torch.is_tensor(output.get("disp")):
        return output["disp"]
    return output


def _valid_disparity_mask(valid, disp_gt, max_disp=None):
    mask = (valid >= 0.5) & torch.isfinite(disp_gt) & (disp_gt >= 0.0)
    if max_disp is not None and max_disp > 0:
        mask = mask & (disp_gt < float(max_disp))
    return mask.reshape(-1)

def _middlebury_region_masks(mask_image, disp_gt, max_disp=None):
    """Build All/Occ/Nocc masks from Middlebury's 0/128/255 mask codes."""
    mask_values = np.asarray(mask_image, dtype=np.uint8)
    mask_codes = torch.from_numpy(mask_values.copy()).to(disp_gt.device).reshape(-1)
    if mask_codes.numel() != disp_gt.numel():
        raise ValueError(
            "Middlebury region mask/GT size mismatch: "
            f"{mask_codes.numel()} vs {disp_gt.numel()}"
        )

    annotated = (mask_codes == 128) | (mask_codes == 255)
    val_all = _valid_disparity_mask(
        annotated.reshape_as(disp_gt), disp_gt, max_disp
    )
    val_occ = val_all & (mask_codes == 128)
    val_nocc = val_all & (mask_codes == 255)
    return val_all, val_occ, val_nocc



def _require_finite_prediction(disp_pr, valid_mask, image_name):
    pred_flat = disp_pr.reshape(-1)
    if valid_mask.numel() != pred_flat.numel():
        print(
            f"{image_name}: prediction/mask size mismatch "
            f"({pred_flat.numel()} vs {valid_mask.numel()})"
        )
    if not bool(valid_mask.any()):
        print(f"{image_name}: no valid ground-truth pixels")

    invalid = valid_mask & ~torch.isfinite(pred_flat)
    if bool(invalid.any()):
        nan_count = int((valid_mask & torch.isnan(pred_flat)).sum().item())
        inf_count = int((valid_mask & torch.isinf(pred_flat)).sum().item())
        raise FloatingPointError(
            f"{image_name}: non-finite disparity on valid pixels: "
            f"nan={nan_count}, inf={inf_count}, "
            f"valid={int(valid_mask.sum().item())}"
        )
def _masked_region_metrics(epe, out, mask, image_name, region):
    if not bool(mask.any()):
        logging.warning("%s: empty Middlebury %s region", image_name, region)
        return float("nan"), float("nan")
    return epe[mask].mean().item(), out[mask].float().mean().item()



def _image_name(data_blob, index):
    return str(data_blob.get("imageL_file", f"image_{index}"))



def save_residual_gray_png_1hw(pred: torch.Tensor,
                               gt: torch.Tensor,
                               png_path: str,
                               abs_residual: bool = True,
                               vmin: float = 0.0,
                               vmax: float = 10.0,
                               invalid_mask: torch.Tensor = None,
                               invalid_value: int = 0):
    """
    pred, gt: torch.Tensor [1, H, W]
    png_path: 输出灰度 PNG 路径（单通道）
    abs_residual: True -> |pred-gt|；False -> pred-gt（会被 clip 到 [vmin,vmax] 再映射）
    vmin, vmax: 线性映射范围：
        vmin -> 0 (黑), vmax -> 255 (白)
        超出范围会 clip
    invalid_mask: 可选，[1,H,W] 或 [H,W] 的 bool/0-1 mask，False/0 表示无效像素
                  无效像素会被写成 invalid_value（默认 0 黑色）
    invalid_value: 0..255，无效像素的灰度值
    返回: residual_map (torch.Tensor [H,W], float32, CPU)
    """
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[0] != 1 or gt.shape[0] != 1:
        raise ValueError(f"Expect [1,H,W], got pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} vs gt={tuple(gt.shape)}")

    res = (pred - gt).squeeze(0).detach().float().cpu()  # [H,W]
    if abs_residual:
        res = res.abs()

    # 处理无效区域（可选）
    if invalid_mask is not None:
        if isinstance(invalid_mask, torch.Tensor):
            m = invalid_mask.detach().cpu()
            if m.ndim == 3:
                m = m.squeeze(0)
            m = m.bool()
        else:
            raise TypeError("invalid_mask must be a torch.Tensor")
    else:
        m = None

    # 线性映射到 0..255
    vmin = float(vmin)
    vmax = float(vmax)
    if not (vmax > vmin):
        raise ValueError(f"Require vmax>vmin, got vmin={vmin}, vmax={vmax}")

    res_np = res.numpy()
    res_clip = np.clip(res_np, vmin, vmax)
    gray = (res_clip - vmin) / (vmax - vmin)  # 0..1
    gray_u8 = (gray * 255.0 + 0.5).astype(np.uint8)

    # 写入无效像素
    if m is not None:
        m_np = m.numpy()
        gray_u8[~m_np] = np.uint8(invalid_value)

    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    Image.fromarray(gray_u8, mode="L").save(png_path)

    return res


@torch.no_grad()
def validate_things(model, iters=32, scale_iters=8, mixed_prec=False,
                    max_disp=1000, bad_threshold=3.0, batch_size=1):
    """Perform validation using the FlyingThings3D TEST split."""
    if batch_size <= 0:
        raise ValueError(f"evaluation batch size must be positive, got {batch_size}")
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    epe_sum = 0.0
    image_count = 0
    outlier_count = 0
    valid_count = 0
    elapsed_sum = 0.0
    elapsed_count = 0
    skipped_count = 0
    indices = list(_validation_indices(len(val_dataset)))
    batch_starts = range(0, len(indices), batch_size)
    for batch_start in tqdm(batch_starts, disable=not _is_main_process()):
        batch_ids = indices[batch_start:batch_start + batch_size]
        data_blobs = [val_dataset[val_id] for val_id in batch_ids]
        image_shapes = {tuple(item["img1"].shape) for item in data_blobs}
        if len(image_shapes) != 1:
            raise ValueError(
                "FlyingThings evaluation batching requires equal image sizes; "
                f"batch indices {batch_ids} have shapes {sorted(image_shapes)}"
            )

        image1 = torch.stack([item["img1"] for item in data_blobs]).cuda()
        image2 = torch.stack([item["img2"] for item in data_blobs]).cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        start = time.time()
        disp_pr = _eval_forward(model, mixed_prec, image1, image2,
                                iters=iters, scale_iters=scale_iters, test_mode=True)
        end = time.time()
        if min(batch_ids) > 50:
            elapsed_sum += end - start
            elapsed_count += len(batch_ids)

        disp_pr_batch = padder.unpad(disp_pr).cpu()
        for batch_index, (val_id, data_blob) in enumerate(zip(batch_ids, data_blobs)):
            disp_pr = disp_pr_batch[batch_index]
            disp_gt = data_blob["disp"]
            valid = data_blob["valid"]
            path = data_blob["imageL_file"]
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0).flatten()
            val = _valid_disparity_mask(valid, disp_gt, max_disp)
            if not bool(val.any()):
                skipped_count += 1
                if _is_main_process():
                    logging.warning(
                        "%s: skipping sample with no valid GT pixels",
                        path,
                    )
                continue
            _require_finite_prediction(disp_pr, val, str(path))

            out = epe > bad_threshold
            image_out = out[val].float().mean().item()
            image_epe = epe[val].mean().item()
            if _is_main_process() and (val_id < 900 or (val_id + 1) % 1000 == 0):
                per_image_runtime = (end - start) / len(batch_ids)
                logging.info(
                    f"FlyingThings3D Iter {val_id+1} out of {len(val_dataset)}. "
                    f"EPE {round(image_epe,4)} Out{bad_threshold} {round(image_out,4)}. "
                    f"Batch {len(batch_ids)} runtime: {format(end-start, '.3f')}s "
                    f"({format(1/per_image_runtime, '.2f')}-images/s/GPU)"
                )

            epe_sum += image_epe
            image_count += 1
            outlier_count += int(out[val].sum().item())
            valid_count += int(val.sum().item())

    (epe_sum, image_count, outlier_count, valid_count,
     elapsed_sum, elapsed_count, skipped_count) = _sum_across_processes([
        epe_sum, image_count, outlier_count, valid_count,
        elapsed_sum, elapsed_count, skipped_count,
    ])
    _require_nonempty_evaluation(image_count, "FlyingThings3D")
    epe = epe_sum / image_count
    out = 100.0 * outlier_count / valid_count
    avg_runtime = elapsed_sum / elapsed_count if elapsed_count > 0 else float("nan")

    if _is_main_process():
        if skipped_count > 0:
            logging.warning(
                "Skipped %d FlyingThings samples with no valid GT pixels",
                int(skipped_count),
            )
        aggregate_fps = _world_size() / avg_runtime
        print(f"Validation FlyingThings: EPE {epe}, Out{bad_threshold} {out}, "
              f"{format(aggregate_fps, '.2f')}-FPS aggregate "
              f"({format(avg_runtime, '.3f')}s/image/GPU)")
    return {'things-epe': epe, 'things-out': out}


@torch.no_grad()
def validate_eth3d(model, iters=32, scale_iters=8, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params, is_eval=True)

    epe_sum = 0.0
    out_sum = 0.0
    image_count = 0
    for val_id in tqdm(_validation_indices(len(val_dataset)), disable=not _is_main_process()):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        disp_pr = _eval_forward(model, mixed_prec, image1, image2,
                                iters=iters, scale_iters=scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe_flattened = epe.flatten()
        val = _valid_disparity_mask(valid, disp_gt)
        _require_finite_prediction(disp_pr, val, _image_name(data_blob, val_id))
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if _is_main_process():
            logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_sum += image_epe
        out_sum += image_out
        image_count += 1

    epe_sum, out_sum, image_count = _sum_across_processes(
        [epe_sum, out_sum, image_count]
    )
    _require_nonempty_evaluation(image_count, "ETH3D")
    epe = epe_sum / image_count
    out1 = 100.0 * out_sum / image_count

    if _is_main_process():
        print("Validation ETH3D: EPE %f, Out1 %f" % (epe, out1))
    return {'eth3d-epe': epe, 'eth3d-out1': out1}


@torch.no_grad()
def validate_kitti(model, iters=32, scale_iters=8, split='15', mixed_prec=False):
    """ Peform validation using the KITTI-2015/2012 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, split=split, image_set='training', is_eval=True)
    torch.backends.cudnn.benchmark = True

    epe_sum = 0.0
    image_count = 0
    outlier_count = 0
    valid_count = 0
    elapsed_sum = 0.0
    elapsed_count = 0
    for val_id in _validation_indices(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        disp_pr = _eval_forward(model, mixed_prec, image1, image2,
                                iters=iters, scale_iters=scale_iters, test_mode=True)
        end = time.time()
        if val_id > 50:
            elapsed_sum += end - start
            elapsed_count += 1

        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe_flattened = epe.flatten()
        val = _valid_disparity_mask(valid, disp_gt)
        _require_finite_prediction(disp_pr, val, _image_name(data_blob, val_id))

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if _is_main_process() and (val_id < 9 or (val_id+1) % 1 == 0):
            logging.info(f"KITTI{split} Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Out3 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_sum += image_epe
        image_count += 1
        outlier_count += int(out[val].sum().item())
        valid_count += int(val.sum().item())

    (epe_sum, image_count, outlier_count, valid_count,
     elapsed_sum, elapsed_count) = _sum_across_processes([
        epe_sum, image_count, outlier_count, valid_count,
        elapsed_sum, elapsed_count,
    ])
    _require_nonempty_evaluation(image_count, f"KITTI{split}")
    epe = epe_sum / image_count
    out3 = 100.0 * outlier_count / valid_count
    avg_runtime = elapsed_sum / elapsed_count if elapsed_count > 0 else float("nan")

    if _is_main_process():
        aggregate_fps = _world_size() / avg_runtime
        print(f"Validation KITTI{split}: EPE {epe}, Out3 {out3}, "
              f"{format(aggregate_fps, '.2f')}-FPS aggregate "
              f"({format(avg_runtime, '.3f')}s/image/GPU)")
    return {f'kitti{split}-epe': epe, f'kitti{split}-out3': out3}


@torch.no_grad()
def validate_middlebury(model, iters=32, scale_iters=8, split='H', mixed_prec=False, eval_max_disp=1000):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, is_eval=True)

    epe_sum = 0.0
    out_sum = 0.0
    image_count = 0
    for val_id in _validation_indices(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        disp_pr = _eval_forward(model, mixed_prec, image1, image2,
                                iters=iters, scale_iters=scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe_flattened = epe.flatten()
        val = _valid_disparity_mask(valid, disp_gt, eval_max_disp)
        _require_finite_prediction(disp_pr, val, _image_name(data_blob, val_id))

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if _is_main_process():
            logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. "
                         f"EPE {round(image_epe,4)} Out2 {round(image_out,4)}")
        epe_sum += image_epe
        out_sum += image_out
        image_count += 1

    epe_sum, out_sum, image_count = _sum_across_processes(
        [epe_sum, out_sum, image_count]
    )
    _require_nonempty_evaluation(image_count, f"Middlebury{split}")
    epe = epe_sum / image_count
    out2 = 100.0 * out_sum / image_count

    if _is_main_process():
        print(f"Validation Middlebury{split}: EPE {epe}, Out2 {out2}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-out2': out2}


def compute_nontexture(x, weight=None, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01, window=33, threshold=0.95, split="F"):

    if split=="H":
        scale = 2
        threshold += 0.02
    elif split=="Q":
        scale = 4
        threshold += 0.03
    else:
        scale = 1
    
    x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
    
    if x.max()>1:
        x = x/x.max()
    
    y = F.pad(x, (1, 1, 1, 1), mode='replicate')
    _, _, h, w = y.shape
    #y = y[..., 0:h-2, 1:w-1] #(y[..., 0:h-2, 1:w-1] + y[..., 2:h, 1:w-1] + y[..., 1:h-1, 0:w-2] + y[..., 1:h-1, 2:w])/4.0

    x = F.pad(x, (window//2, window//2, window//2, window//2), mode='replicate')
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError(
            'Both c1 and c2 are infinite, SSIM loss is zero. This is '
            'likely unintended.')
    _, _, H, W = x.shape

    if weight is None:
        weight = torch.ones((H, W)).to(x)
    else:
        assert weight.shape == (H, W), \
                f'image shape is {(H, W)}, but weight shape is {weight.shape}'
    weight = weight[None, None, ...]
    average_pooled_weight = F.avg_pool2d(weight, (window, window), stride=(1, 1))
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (
        average_pooled_weight + weight_epsilon)

    def weighted_avg_pool(z):
        weighted_avg = F.avg_pool2d(
            z * weight_plus_epsilon, (window, window), stride=(1, 1))
        return weighted_avg * inverse_average_pooled_weight
    
    mu_x = weighted_avg_pool(x)
    sigma_x = weighted_avg_pool(x**2) - mu_x**2

    def ssim(x, y):
        y = F.pad(y, (window//2, window//2, window//2, window//2), mode='replicate')
        mu_y = weighted_avg_pool(y)
        sigma_y = weighted_avg_pool(y**2) - mu_y**2
        sigma_xy = weighted_avg_pool(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x**2 + mu_y**2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

            result = ssim_n / ssim_d

        result = F.avg_pool2d(result, (scale, scale), stride=(scale, scale))

        return result
    
    mask = (ssim(x, y[..., 0:h-2, 1:w-1])>threshold) & (ssim(x, y[..., 2:h, 1:w-1])>threshold) & (ssim(x, y[..., 1:h-1, 0:w-2])>threshold) & (ssim(x, y[..., 1:h-1, 2:w])>threshold)
    mask = mask[0, 0] & mask[0, 1] & mask[0, 2]
    
    return mask.cpu().numpy()


@torch.no_grad()
def validate_middlebury_indetail(model, iters=32, scale_iters=8, split='H', mixed_prec=False, eval_max_disp=1000):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, is_eval=True)

    out_list, epe_list, portion_list = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for val_id in _validation_indices(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        #torch.save(disp_gt, "gt.pth")
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        disp_pr = _eval_forward(model, mixed_prec, image1, image2,
                                iters=iters, scale_iters=scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)
        
        epe_flattened = epe.flatten()
        
        occ_mask = Image.open(data_blob["imageL_file"].replace('im0.png', 'mask0nocc.png')).convert('L')
        val_all, val_occ, val_nocc = _middlebury_region_masks(
            occ_mask, disp_gt, eval_max_disp
        )
        _require_finite_prediction(disp_pr, val_all, _image_name(data_blob, val_id))

        nontexture_mask = torch.as_tensor(
            compute_nontexture(data_blob["img1"][None].cuda(), split=split),
            dtype=torch.bool,
            device=val_all.device,
        ).reshape(-1)
        val_ntt = val_all & nontexture_mask
        
        out = (epe_flattened > 2.0)
        image_out = out[val_all].float().mean().item()
        image_epe = epe_flattened[val_all].mean().item()

        name = _image_name(data_blob, val_id)
        image_epe_occ, image_out_occ = _masked_region_metrics(
            epe_flattened, out, val_occ, name, "occ")
        image_epe_nocc, image_out_nocc = _masked_region_metrics(
            epe_flattened, out, val_nocc, name, "nocc")
        image_epe_ntt, image_out_ntt = _masked_region_metrics(
            epe_flattened, out, val_ntt, name, "nontexture")

        if _is_main_process():
            logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. "
                         f"All({round((val_all.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe,4)} Out2 {round(image_out,4)}, \n "
                         f"Occ({round((val_occ.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe_occ,4)} Out2 {round(image_out_occ,4)}, "
                         f"NOcc({round((val_nocc.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe_nocc,4)} Out2 {round(image_out_nocc,4)}, "
                         f"NonTexture({round((val_ntt.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe_ntt,4)} Out2 {round(image_out_ntt,4)}")
        
        epe_list[0].append(image_epe)
        out_list[0].append(image_out)
        portion_list[0].append((val_all.sum()/val_all.sum()).item())
        epe_list[1].append(image_epe_occ)
        out_list[1].append(image_out_occ)
        portion_list[1].append((val_occ.sum()/val_all.sum()).item())
        epe_list[2].append(image_epe_nocc)
        out_list[2].append(image_out_nocc)
        portion_list[2].append((val_nocc.sum()/val_all.sum()).item())
        epe_list[3].append(image_epe_ntt)
        out_list[3].append(image_out_ntt)
        portion_list[3].append((val_ntt.sum()/val_all.sum()).item())

    epe = _distributed_nanmean_rows(epe_list)
    out2 = 100.0 * _distributed_nanmean_rows(out_list)
    portion = 100.0 * _distributed_nanmean_rows(portion_list)
    _require_nonempty_evaluation(
        _sum_across_processes([len(epe_list[0])])[0],
        f"Middlebury{split}",
    )

    if _is_main_process():
        print(f"Validation Middlebury{split}: All({round(portion[0],8)}%): EPE {round(epe[0],8)} Out2 {round(out2[0],8)}, \n"
                         f"Occ({round(portion[1],8)}%): EPE {round(epe[1],8)} Out2 {round(out2[1],8)}, "
                         f"NOcc({round(portion[2],8)}%): EPE {round(epe[2],8)} Out2 {round(out2[2],8)}, "
                         f"NonTexture({round(portion[3],8)}%): EPE {round(epe[3],8)} Out2 {round(out2[3],8)}")
    return {f'middlebury{split}-epe': epe[0], f'middlebury{split}-out2': out2[0]}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['cor_ga', 'pact', 'pact_smd', 'pact_smd_post', 'pact_bilap_gru', 'pact2', 'pact2_gev', 'pact_pivno', 'defom_pivno', 'defom_pivno_gated'],
        default='cor_ga',
        help="model family; PACT is opt-in to preserve old checkpoint loading",
    )
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--datasets', nargs='+', type=str, help="dataset for evaluation", default=["things"],
                        choices=["things", "eth3d", "kitti12", "kitti15"] + [f"middlebury_{s}" for s in ['F','H','Q','2021']])
    parser.add_argument('--indetail', action='store_true', help='evaluate middlebury in detail (for different regions)')
    parser.add_argument('--max_disp', type=int, default=None,
                        help='model search range; required for PACT checkpoints')
    parser.add_argument('--eval_max_disp', type=float, default=1000.0,
                        help='independent upper bound for valid GT pixels; <=0 disables the bound')
    parser.add_argument('--pact_debug_finite', action='store_true')
    parser.add_argument('--pact_mid_refine_iters', type=int, default=1)
    parser.add_argument(
        '--pact_smd_stage', choices=['head', 'joint', 'full'], default='joint'
    )
    parser.add_argument('--pact_smd_grad_iters', type=int, default=2)
    parser.add_argument('--pact_smd_mode_threshold', type=float, default=0.5)
    parser.add_argument('--pact_smd_post_mode_threshold', type=float, default=0.5)
    parser.add_argument('--pact_smd_post_local_radius', type=float, default=1.0)
    parser.add_argument('--pact_smd_post_broad_radius_min', type=float, default=2.0)
    parser.add_argument('--pact_smd_post_broad_radius_max', type=float, default=32.0)
    parser.add_argument('--bilap_ablation', choices=['single_laplace', 'dual_no_interaction', 'dual_symmetric_interaction'], default='dual_symmetric_interaction')
    parser.add_argument('--bilap_init', choices=['smd', 'symmetric'], default='smd')
    parser.add_argument('--bilap_init_delta', type=float, default=2.0)
    parser.add_argument('--bilap_init_scale', type=float, default=2.0)
    parser.add_argument('--bilap_separate_mode_gru', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bilap_lookup_mode', choices=['fixed', 'scale_aware'], default='scale_aware')
    parser.add_argument('--bilap_q_min', type=float, default=1.0)
    parser.add_argument('--bilap_q_max', type=float, default=4.0)
    parser.add_argument('--bilap_q_scale', type=float, default=0.5)
    parser.add_argument('--pact_gev_mode', choices=['coarse', 'dual'], default='dual')
    parser.add_argument('--pact_sampling_layout', choices=['legacy9', 'wide9'], default='legacy9')
    parser.add_argument('--pact_min_radius', type=float, default=1.0)
    parser.add_argument('--pact_max_radius', type=float, default=8.0)
    parser.add_argument('--pact_mid_delta_scale', type=float, default=1.0)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disparity field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=20, help="number of scaling updates to the disparity field in each forward pass.")
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='per-GPU batch size for FlyingThings evaluation')

    # Architecure choices
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--distributed', action='store_true',
                        help='shard evaluation samples across torchrun processes')

    args = parser.parse_args()
    pact_model = args.model in ('pact', 'pact_smd', 'pact_smd_post', 'pact_bilap_gru', 'pact2', 'pact2_gev', 'pact_pivno', 'defom_pivno', 'defom_pivno_gated')
    if args.max_disp is None:
        if pact_model:
            parser.error('--max_disp is required for PACT because legacy checkpoints do not store it')
        args.max_disp = 200

    # Restore architecture metadata before model construction. Candidate
    # layout changes tensor semantics even when state-dict shapes agree.
    if pact_model and args.restore_ckpt is not None:
        architecture_checkpoint = torch.load(args.restore_ckpt, map_location='cpu')
        if isinstance(architecture_checkpoint, dict):
            if args.model in ('pact_pivno', 'defom_pivno', 'defom_pivno_gated'):
                args.pivno_input_channels = _infer_pivno_input_channels(
                    architecture_checkpoint
                )
            architecture_config = architecture_checkpoint.get('model_config')
            if isinstance(architecture_config, dict):
                restore_fields = [
                    ('pact_sampling_layout', 'legacy9'),
                    ('pact_mid_refine_iters', 1),
                    ('idepth_scale', 0.5),
                ]
                if args.model == 'pact2_gev':
                    restore_fields.append(('pact_gev_mode', 'dual'))
                if args.model in ('pact', 'pact_smd', 'pact_smd_post', 'pact_bilap_gru'):
                    restore_fields.extend([
                        ('pact_min_radius', 1.0),
                        ('pact_max_radius', 8.0),
                        ('pact_mid_delta_scale', 1.0),
                    ])
                if args.model == 'pact_smd':
                    restore_fields.extend([
                        ('pact_smd_stage', 'joint'),
                        ('pact_smd_grad_iters', 2),
                        ('pact_smd_mode_threshold', 0.5),
                    ])
                if args.model == 'pact_smd_post':
                    restore_fields.extend([
                        ('pact_smd_mode_threshold', 0.5),
                        ('pact_smd_post_mode_threshold', 0.5),
                        ('pact_smd_post_local_radius', 1.0),
                        ('pact_smd_post_broad_radius_min', 2.0),
                        ('pact_smd_post_broad_radius_max', 32.0),
                    ])
                if args.model == 'pact_bilap_gru':
                    restore_fields.extend([
                        ('pact_smd_mode_threshold', 0.5),
                        ('bilap_ablation', 'dual_symmetric_interaction'),
                        ('bilap_init', 'smd'),
                        ('bilap_init_delta', 2.0),
                        ('bilap_init_scale', 2.0),
                        ('bilap_separate_mode_gru', False),
                        ('bilap_lookup_mode', 'scale_aware'),
                        ('bilap_q_min', 1.0),
                        ('bilap_q_max', 4.0),
                        ('bilap_q_scale', 0.5),
                    ])
                for key, default in restore_fields:
                    setattr(args, key, architecture_config.get(key, default))

    if args.distributed:
        if not torch.cuda.is_available():
            parser.error('--distributed evaluation requires CUDA')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        local_rank = 0
        torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.empty_cache()
    if args.distributed:
        print(
            f"Distributed evaluator rank {_rank()}/{_world_size()} "
            f"using cuda:{local_rank}",
            flush=True,
        )

    if args.model == 'pact_bilap_gru':
        model_cls = PACTBiLapGRUDEFOMStereo
    elif args.model == 'pact_smd_post':
        model_cls = PACTSMDPostDEFOMStereo
    elif args.model == 'pact_smd':
        model_cls = PACTSMDDEFOMStereo
    elif args.model == 'pact2_gev':
        model_cls = PACT2GEVDEFOMStereo
    elif args.model == 'pact2':
        model_cls = PACT2DEFOMStereo
    elif args.model == 'pact':
        model_cls = PACTDEFOMStereo
    elif args.model == 'pact_pivno':
        model_cls = PACTPIVNODEFOMStereo
    elif args.model == 'defom_pivno':
        model_cls = PIVNODEFOMStereo
    elif args.model == 'defom_pivno_gated':
        model_cls = GatedPIVNODEFOMStereo
    else:
        if CorGADEFOMStereo is None:
            parser.error(
                "--model cor_ga requires core/defom_cor_ga.py, which is not "
                "present in this portable checkout"
            )
        model_cls = CorGADEFOMStereo
    model = model_cls(args)

    logging.basicConfig(level=logging.INFO if _is_main_process() else logging.ERROR,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    logging.info(
        "Evaluation config: model=%s model_max_disp=%s eval_max_disp=%s amp=%s "
        "amp_owner=%s world_size=%s things_batch_size_per_gpu=%s "
        "things_global_batch_size=%s",
        args.model, args.max_disp, args.eval_max_disp, args.mixed_precision,
        "model" if _model_manages_amp(model) else "evaluator",
        _world_size(),
        args.eval_batch_size,
        args.eval_batch_size * _world_size(),
    )
    if any(dataset != 'things' for dataset in args.datasets):
        logging.info("Non-FlyingThings validators use per-GPU batch size 1")
    if args.restore_ckpt is not None:
        if not args.restore_ckpt.endswith((".pth", ".pth.gz")):
            raise ValueError(
                "--restore_ckpt must point to a torch checkpoint ending in "
                "'.pth' or the training script's legacy '.pth.gz' suffix"
            )
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location=device)
        if pact_model:
            if not isinstance(checkpoint, dict) or 'model_config' not in checkpoint:
                raise ValueError(
                    "Anchor-free PACT checkpoints require model_config metadata; "
                    "legacy anchor checkpoints are not compatible"
                )
            config = checkpoint['model_config']
            checkpoint_max_disp = config.get('max_disp')
            requested_max_disp = int(args.max_disp)
            if checkpoint_max_disp != requested_max_disp:
                logging.warning(
                    "Overriding checkpoint max_disp=%s with evaluation model "
                    "max_disp=%s. This changes the inference search range and "
                    "should be treated as an extrapolation experiment.",
                    checkpoint_max_disp,
                    requested_max_disp,
                )
            expected = {
                'n_downsample': int(args.n_downsample),
                'n_gru_layers': int(args.n_gru_layers),
                'hidden_dims': list(args.hidden_dims),
            }
            if args.model in ('pact_pivno', 'defom_pivno', 'defom_pivno_gated'):
                expected.update({
                    'model': args.model,
                    'model_variant': args.model,
                    'state_mode': 'pivno_single_current_disp',
                })
                if args.model == 'defom_pivno_gated':
                    expected['pivno_scale_gate'] = model_cls.SCALE_GATE_MODE
                    expected['corr_radius'] = int(args.corr_radius)
                if 'pivno_input_channels' in config:
                    expected['pivno_input_channels'] = int(
                        args.pivno_input_channels
                    )
            elif args.model in ('pact2', 'pact2_gev'):
                expected['dinov2_encoder'] = args.dinov2_encoder
                expected['state_mode'] = 'single_current_disp'
                expected.update({
                    'model': args.model,
                    'model_variant': model_cls.MODEL_VARIANT,
                    'feature_backbone': model_cls.FEATURE_BACKBONE,
                    'idepth_scale': float(args.idepth_scale),
                    'pact_mid_refine_iters': int(args.pact_mid_refine_iters),
                    'pact_sampling_layout': model_cls.SAMPLING_LAYOUT,
                    'pact_fixed_radius': model_cls.FIXED_RADIUS_QUARTER,
                })
                if args.model == 'pact2_gev':
                    expected['pact_gev_mode'] = str(args.pact_gev_mode)
            else:
                expected['dinov2_encoder'] = args.dinov2_encoder
                expected['state_mode'] = (
                    'dual_laplace_distribution'
                    if args.model == 'pact_bilap_gru'
                    else 'single_current_disp'
                )
                expected.update({
                    'pact_sampling_layout': str(args.pact_sampling_layout),
                    'pact_min_radius': float(args.pact_min_radius),
                    'pact_max_radius': float(args.pact_max_radius),
                    'pact_mid_delta_scale': float(args.pact_mid_delta_scale),
                })
                if args.model == 'pact_smd':
                    expected.update({
                        'model': 'pact_smd',
                        'model_variant': model_cls.MODEL_VARIANT,
                        'pact_smd_mode_threshold': float(
                            args.pact_smd_mode_threshold
                        ),
                    })
                if args.model == 'pact_smd_post':
                    expected.update({
                        'model': 'pact_smd_post',
                        'model_variant': model_cls.MODEL_VARIANT,
                        'pact_smd_stage': 'post',
                        'pact_smd_mode_threshold': float(
                            args.pact_smd_mode_threshold
                        ),
                        'pact_smd_post_mode_threshold': float(
                            args.pact_smd_post_mode_threshold
                        ),
                        'pact_smd_post_local_radius': float(
                            args.pact_smd_post_local_radius
                        ),
                        'pact_smd_post_broad_radius_min': float(
                            args.pact_smd_post_broad_radius_min
                        ),
                        'pact_smd_post_broad_radius_max': float(
                            args.pact_smd_post_broad_radius_max
                        ),
                    })
                if args.model == 'pact_bilap_gru':
                    expected.update({
                        'model': 'pact_bilap_gru',
                        'model_variant': model_cls.MODEL_VARIANT,
                        'bilap_width_levels': 2,
                        'bilap_match_channels': 32,
                        'bilap_gwc_groups': 4,
                        'pact_smd_mode_threshold': float(args.pact_smd_mode_threshold),
                        'bilap_ablation': str(args.bilap_ablation),
                        'bilap_init': str(args.bilap_init),
                        'bilap_init_delta': float(args.bilap_init_delta),
                        'bilap_init_scale': float(args.bilap_init_scale),
                        'bilap_separate_mode_gru': bool(args.bilap_separate_mode_gru),
                        'bilap_lookup_mode': str(args.bilap_lookup_mode),
                        'bilap_q_min': float(args.bilap_q_min),
                        'bilap_q_max': float(args.bilap_q_max),
                        'bilap_q_scale': float(args.bilap_q_scale),
                    })
            mismatches = {
                key: (
                    config.get(
                        key,
                        {
                            'pact_sampling_layout': 'legacy9',
                            'pact_min_radius': 1.0,
                            'pact_max_radius': 8.0,
                            'pact_mid_delta_scale': 1.0,
                        }.get(key),
                    ),
                    value,
                )
                for key, value in expected.items()
                if config.get(
                    key,
                    {
                        'pact_sampling_layout': 'legacy9',
                        'pact_min_radius': 1.0,
                        'pact_max_radius': 8.0,
                        'pact_mid_delta_scale': 1.0,
                    }.get(key),
                ) != value
            }
            if mismatches:
                raise ValueError(
                    f"checkpoint/model configuration mismatch: {mismatches}"
                )
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logging.info(f"Done loading checkpoint")

    model.to(device)
    model.eval()

    if _is_main_process():
        print(f"The model has {format(count_parameters(model)[1]/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = (
        (pact_model and args.mixed_precision)
        or args.corr_implementation.endswith("_cuda")
    )

    if 'things' in args.datasets:
        validate_things(
            model,
            iters=args.valid_iters,
            scale_iters=args.scale_iters,
            mixed_prec=use_mixed_precision,
            max_disp=args.eval_max_disp if args.eval_max_disp > 0 else args.max_disp,
            batch_size=args.eval_batch_size,
        )

    if 'eth3d' in args.datasets:
        validate_eth3d(model, iters=args.valid_iters, scale_iters=args.scale_iters, mixed_prec=use_mixed_precision)

    if 'kitti12' in args.datasets:
        validate_kitti(model, iters=args.valid_iters, scale_iters=args.scale_iters, split='12', mixed_prec=use_mixed_precision)

    if 'kitti15' in args.datasets:
        validate_kitti(model, iters=args.valid_iters, scale_iters=args.scale_iters, split='15', mixed_prec=use_mixed_precision)

    for s in ['F', 'H', 'Q', '2021']:
        if f"middlebury_{s}" in args.datasets:
            if args.indetail:
                validate_middlebury_indetail(model, iters=args.valid_iters, scale_iters=args.scale_iters, split=s,
                                             mixed_prec=use_mixed_precision, eval_max_disp=args.eval_max_disp)
            else:
                validate_middlebury(model, iters=args.valid_iters, scale_iters=args.scale_iters, split=s,
                                    mixed_prec=use_mixed_precision, eval_max_disp=args.eval_max_disp)

    if _distributed_enabled():
        dist.barrier()
        dist.destroy_process_group()
