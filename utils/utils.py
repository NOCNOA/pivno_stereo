import logging
import math
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random

def normalize_disparity(disp):
    dtype = disp.dtype
    scale, shift = compute_scale_and_shift(disp.float())
    norm_disp = (disp - shift[..., None, None]) / scale[..., None, None]

    return norm_disp.to(dtype), scale.to(dtype), shift.to(dtype)

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def sequence_loss_d0L1(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=1000, smooth_l1_beta=1.0):
    """ 
    Loss function defined over sequence of flow predictions.

    flow_preds[0] is treated as initial disparity d0 and supervised by SmoothL1.
    flow_preds[1:] keep the original RAFT-style L1 sequence loss.
    """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # same adjusted gamma logic as original RAFT-Stereo
    adjusted_loss_gamma = loss_gamma ** (15 / n_predictions)

    for i in range(n_predictions):
        pred = flow_preds[i]

        assert not torch.isnan(pred).any() and not torch.isinf(pred).any()

        i_weight = adjusted_loss_gamma ** (n_predictions - i)

        if i == 0:
            # d0: SmoothL1 loss
            i_loss = F.smooth_l1_loss(
                pred,
                flow_gt,
                reduction='none',
                beta=smooth_l1_beta
            )
        else:
            # other disparity predictions: original L1 loss
            i_loss = (pred - flow_gt).abs()

        assert i_loss.shape == valid.shape, [
            i_loss.shape, valid.shape, flow_gt.shape, pred.shape
        ]

        flow_loss += i_weight * i_loss[valid.bool()].mean()

    # metrics keep unchanged, only evaluate final prediction
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def sequence_loss_withconf(flow_preds, conf_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    B, _, H, W = flow_gt.shape
    #print("asd", flow_gt.shape)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        
        
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # with torch.no_grad():
        #     conf_gt = (i_loss <= 2).float()

        disp_loss = i_loss 
        #loss_conf_map = F.binary_cross_entropy_with_logits(logit_preds[i], conf_gt, reduction='none')
        w = (1 - conf_preds[i]).detach()
        w = w / (w.mean(dim=[2,3], keepdim=True) + 1e-6)
        w = w.clamp(0.25, 4.0)

        final_loss = disp_loss * w
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += (i_weight * final_loss)[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def gradient_x(img):
    return img[..., :, 1:] - img[..., :, :-1]

def gradient_y(img):
    return img[..., 1:, :] - img[..., :-1, :]

def edge_aware_smoothness(disp, image, alpha=10.0):
    """
    disp:  [B,1,H,W]
    image: [B,3,H,W] or [B,1,H,W]  (用于边缘权重)
    """
    # image 转灰度也行，这里简单用均值
    if image.shape[1] > 1:
        img = image.mean(dim=1, keepdim=True)
    else:
        img = image

    dx_disp = gradient_x(disp)
    dy_disp = gradient_y(disp)
    dx_img  = gradient_x(img).abs()
    dy_img  = gradient_y(img).abs()

    wx = torch.exp(-alpha * dx_img)
    wy = torch.exp(-alpha * dy_img)

    # charbonnier 比 L1 更平滑
    smooth = (wx * charbonnier(dx_disp)).mean() + (wy * charbonnier(dy_disp)).mean()
    return smooth

def sequence_loss_smoothness(
    flow_preds, flow_gt, valid,
    loss_gamma=0.9, max_flow=700,

    # ---- 新增：两类正则的权重 ----
    lambda_step=0.01,      # 建议 0.01 起
    lambda_osc=0.05,       # 建议 0.05 起
    lambda_smooth=0.01,    # 建议 0.001~0.01（需要 image 才能用）
    image=None,           # 传入左图 [B,3,H,W] 才启用 edge-aware smoothness
):
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # gamma 调整保持你原逻辑
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))

    prev_delta = None
    for i in range(n_predictions):
        pred = flow_preds[i]
        assert not torch.isnan(pred).any() and not torch.isinf(pred).any()

        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (pred - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, pred.shape]
        flow_loss = flow_loss + i_weight * i_loss[valid.bool()].mean()

        # --- (1) 步长正则：抑制每步更新过大（不需要额外输出） ---
        if lambda_step > 0.0:
            if i == 0:
                delta = pred - pred.detach()  # 0（第一步没有上一步；你也可以跳过 i==0）
            else:
                delta = pred - flow_preds[i-1].detach()  # 用 detach 防止“互相拉扯”
            # 只在 valid 上算，避免无效区影响
            flow_loss = flow_loss + lambda_step * delta.abs()[valid.bool()].mean()

            # --- (2) 振荡正则：抑制来回震荡 Δt-Δt-1 ---
            if lambda_osc > 0.0 and prev_delta is not None:
                osc = (delta - prev_delta).abs()
                flow_loss = flow_loss + lambda_osc * osc[valid.bool()].mean()
            prev_delta = delta

        # --- (3) 边缘感知平滑：需要 image ---
        if lambda_smooth > 0.0 and image is not None:
            # pred 可能是 flow [B,2,H,W]，你若是 disparity 就是 [B,1,H,W]
            # stereo disparity 情况：只取 x 分量或直接 pred
            if pred.shape[1] == 2:
                disp = pred[:, :1]  # 只用 x 分量当作 disparity
            else:
                disp = pred
            flow_loss = flow_loss + (lambda_smooth * i_weight) * edge_aware_smoothness(disp, image)

    # metrics（保持不动）
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def fetch_optimizer(args, model, last_epoch=-1, checkpoint=None):
    """ Create the optimizer and learning rate scheduler """
    model_without_wrapper = model.module if hasattr(model, 'module') else model
    group_builder = getattr(
        model_without_wrapper, 'optimizer_parameter_groups', None
    )
    if group_builder is None:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_groups = trainable_params
        scheduler_max_lr = args.lr
    else:
        optimizer_groups = group_builder(args)
        if not optimizer_groups or any(not group['params'] for group in optimizer_groups):
            raise ValueError('model returned an empty optimizer parameter group')
        scheduler_max_lr = [float(group['lr']) for group in optimizer_groups]
    optimizer = optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=1e-8,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler_state = (
        checkpoint.get('scheduler') if checkpoint is not None else None
    )
    saved_lrs = [group['lr'] for group in optimizer.param_groups]
    if scheduler_state is not None:
        scheduler_epoch = int(scheduler_state['last_epoch'])
    elif checkpoint is not None:
        optimizer_steps = []
        for state in optimizer.state.values():
            step = state.get('step')
            if step is not None:
                optimizer_steps.append(
                    int(step.item()) if torch.is_tensor(step) else int(step)
                )
        if optimizer_steps:
            scheduler_epoch = max(optimizer_steps)
            if min(optimizer_steps) != scheduler_epoch:
                logging.warning(
                    "Optimizer parameter steps disagree while rebuilding the "
                    "scheduler; using the largest value %d",
                    scheduler_epoch,
                )
        else:
            scheduler_epoch = int(last_epoch)
    else:
        scheduler_epoch = int(last_epoch)

    # LRScheduler performs one initial step in its constructor. Passing N-1
    # therefore restores a scheduler that has completed exactly N updates.
    scheduler_constructor_epoch = (
        scheduler_epoch - 1 if scheduler_epoch >= 0 else -1
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        scheduler_max_lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy='linear',
        last_epoch=scheduler_constructor_epoch,
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        for param_group, lr in zip(
            optimizer.param_groups, scheduler.get_last_lr()
        ):
            param_group['lr'] = lr
    elif checkpoint is not None:
        # Preserve the exact checkpoint LR while using the optimizer update
        # count (not global batches) as the scheduler position. AMP can skip
        # optimizer/scheduler steps when gradients overflow.
        scheduler._last_lr = list(saved_lrs)
        scheduler._step_count = max(1, scheduler_epoch + 1)
        for param_group, lr in zip(optimizer.param_groups, saved_lrs):
            param_group['lr'] = lr

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = 'runs/' + name
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar("train/" + k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar("valid/" + key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def _as_disp_4d(x):
    """Convert disparity-like tensors to [B, 1, H, W]."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected disparity tensor with 3 or 4 dims, got shape {tuple(x.shape)}")

def disparity_gradient_loss(pred, disp_gt, valid):
    """Gradient consistency loss on adjacent valid GT pixels."""
    pred = _as_disp_4d(pred)
    disp_gt = _as_disp_4d(disp_gt).to(dtype=pred.dtype)
    valid = _as_disp_4d(valid).bool()

    total = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)

    if pred.shape[-1] > 1:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = disp_gt[:, :, :, 1:] - disp_gt[:, :, :, :-1]
        valid_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        loss_x = (pred_dx - gt_dx).abs() * valid_x.float()
        total = total + loss_x.sum()
        count = count + valid_x.float().sum()

    if pred.shape[-2] > 1:
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = disp_gt[:, :, 1:, :] - disp_gt[:, :, :-1, :]
        valid_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        loss_y = (pred_dy - gt_dy).abs() * valid_y.float()
        total = total + loss_y.sum()
        count = count + valid_y.float().sum()

    return total / count.clamp_min(1.0)

def disparity_edge_mask(
    disp_gt,
    valid,
    edge_mode="topk",
    edge_topk=0.10,
    edge_threshold=1.0,
    edge_dilation=5,
):
    """Build a GT-disparity edge mask from disparity gradients.

    Returns:
        edge_mask: float tensor in [0, 1], shape [B, 1, H, W]
        edge_ratio: scalar tensor, edge pixels over valid pixels after dilation
        grad: raw gradient strength, shape [B, 1, H, W]
    """
    disp_gt = _as_disp_4d(disp_gt).float()
    valid = _as_disp_4d(valid).bool()

    b, _, h, w = disp_gt.shape
    grad = disp_gt.new_zeros((b, 1, h, w))
    grad_valid = torch.zeros((b, 1, h, w), device=disp_gt.device, dtype=torch.bool)

    if w > 1:
        grad_x = (disp_gt[:, :, :, 1:] - disp_gt[:, :, :, :-1]).abs()
        valid_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        grad[:, :, :, :-1] += grad_x * valid_x.float()
        grad_valid[:, :, :, :-1] |= valid_x

    if h > 1:
        grad_y = (disp_gt[:, :, 1:, :] - disp_gt[:, :, :-1, :]).abs()
        valid_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        grad[:, :, :-1, :] += grad_y * valid_y.float()
        grad_valid[:, :, :-1, :] |= valid_y

    if edge_mode == "threshold":
        edge_mask = (grad > edge_threshold) & grad_valid
    elif edge_mode == "topk":
        edge_mask = torch.zeros_like(grad_valid)
        edge_topk = max(0.0, min(float(edge_topk), 1.0))

        for batch_idx in range(b):
            image_valid = grad_valid[batch_idx, 0]
            values = grad[batch_idx, 0][image_valid]
            if values.numel() == 0 or edge_topk <= 0.0:
                continue

            k = max(1, int(torch.ceil(values.new_tensor(values.numel() * edge_topk)).item()))
            k = min(k, values.numel())
            topk_indices = torch.topk(values, k, largest=True).indices
            valid_indices = image_valid.nonzero(as_tuple=False)
            selected = valid_indices[topk_indices]
            edge_mask[batch_idx, 0, selected[:, 0], selected[:, 1]] = True
    else:
        raise ValueError(f"Unsupported edge_mode '{edge_mode}', expected 'topk' or 'threshold'")

    edge_mask = edge_mask.float()

    if edge_dilation and edge_dilation > 1:
        kernel_size = int(edge_dilation)
        padding = kernel_size // 2
        edge_mask = F.max_pool2d(edge_mask, kernel_size=kernel_size, stride=1, padding=padding)
        if kernel_size % 2 == 0:
            edge_mask = edge_mask[:, :, :h, :w]

    edge_mask = edge_mask * valid.float()
    valid_count = valid.float().sum().clamp_min(1.0)
    edge_ratio = edge_mask.sum() / valid_count

    return edge_mask, edge_ratio, grad


def _mixlap_nll(disp_pred, delta_info, target):
    """Per-pixel two-component Laplace mixture negative log likelihood."""
    disp_pred = _as_disp_4d(disp_pred).float()
    target = _as_disp_4d(target).float()
    if delta_info.ndim != 4 or delta_info.shape[1] != 3:
        raise ValueError(
            "MOL information must be [B,3,H,W], got "
            f"{tuple(delta_info.shape)}"
        )
    if delta_info.shape[0] != disp_pred.shape[0] or \
            delta_info.shape[-2:] != disp_pred.shape[-2:]:
        raise ValueError(
            "MOL information must match disparity batch/spatial shape: "
            f"disp={tuple(disp_pred.shape)}, info={tuple(delta_info.shape)}"
        )

    info = torch.nan_to_num(
        delta_info.float(), nan=0.0, posinf=10.0, neginf=-10.0
    )
    mixture_logits = info[:, :2]
    log_b = info[:, 2:3].clamp(0.0, 10.0)
    error = (target - disp_pred).abs()
    broad_term = error * torch.exp(-log_b) + log_b + math.log(2.0)
    narrow_term = error + math.log(2.0)
    component_terms = torch.cat((broad_term, narrow_term), dim=1)
    return (
        torch.logsumexp(mixture_logits, dim=1, keepdim=True)
        - torch.logsumexp(mixture_logits - component_terms, dim=1, keepdim=True)
    )


def sequence_loss_d0L1_edge(
    flow_preds,
    flow_gt,
    valid,
    loss_gamma=0.9,
    max_flow=768,
    smooth_l1_beta=1.0,
    use_edge_weight_loss=True,
    edge_mode="topk",
    edge_topk=0.10,
    edge_threshold=1.0,
    edge_dilation=5,
    lambda_edge=2.0,
    use_disp_grad_loss=True,
    lambda_grad=0.05,
    use_non_degrade_loss=True,
    lambda_non_degrade=0.02,
    non_degrade_good_px=3.0,
    non_degrade_margin=0.5,
    delta_info_preds=None,
):
    """D0 SmoothL1 plus edge-aware recurrent MixLap or plain L1 loss."""
    if not isinstance(flow_preds, (list, tuple)):
        flow_preds = [flow_preds]

    flow_gt = _as_disp_4d(flow_gt)
    valid = _as_disp_4d(valid)
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    if delta_info_preds is not None:
        if not isinstance(delta_info_preds, (list, tuple)):
            raise TypeError("delta_info_preds must be a list or tuple")
        if len(delta_info_preds) != n_predictions - 1:
            raise ValueError(
                "MOL predictions must align with recurrent disparities: "
                f"got {len(delta_info_preds)} info maps for "
                f"{n_predictions - 1} recurrent predictions"
            )

    finite_gt = torch.isfinite(flow_gt)
    safe_flow_gt = torch.where(
        finite_gt, flow_gt, torch.zeros_like(flow_gt)
    )
    mag = torch.sum(safe_flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid = ((valid >= 0.5) & finite_gt & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]

    if valid.any():
        assert torch.isfinite(safe_flow_gt[valid.bool()]).all()

    edge_mask, edge_ratio, _ = disparity_edge_mask(
        safe_flow_gt,
        valid,
        edge_mode=edge_mode,
        edge_topk=edge_topk,
        edge_threshold=edge_threshold,
        edge_dilation=edge_dilation,
    )

    if use_edge_weight_loss:
        loss_weight = 1.0 + float(lambda_edge) * edge_mask
    else:
        loss_weight = torch.ones_like(edge_mask)
        edge_mask = torch.zeros_like(edge_mask)
        edge_ratio = edge_ratio.detach() * 0.0

    valid_float = valid.float()
    raw_sequence_loss = safe_flow_gt.new_tensor(0.0)
    weighted_sequence_loss = safe_flow_gt.new_tensor(0.0)
    adjusted_loss_gamma = loss_gamma ** (15 / n_predictions)

    for i, pred in enumerate(flow_preds):
        pred = _as_disp_4d(pred)
        finite_pred = torch.isfinite(pred)
        if not bool(finite_pred.all()):
            finite_values = pred.detach()[finite_pred]
            value_range = (
                "no finite values"
                if finite_values.numel() == 0
                else f"finite_range=[{finite_values.min().item():.6g}, "
                     f"{finite_values.max().item():.6g}]"
            )
            raise FloatingPointError(
                f"non-finite disparity prediction[{i}] "
                f"shape={tuple(pred.shape)}, "
                f"nan={torch.isnan(pred).sum().item()}, "
                f"inf={torch.isinf(pred).sum().item()}, {value_range}"
            )

        i_weight = adjusted_loss_gamma ** (n_predictions - i)

        if i == 0:
            i_loss = F.smooth_l1_loss(
                pred, safe_flow_gt, reduction='none', beta=smooth_l1_beta
            )
        elif delta_info_preds is not None:
            i_loss = _mixlap_nll(
                pred, delta_info_preds[i - 1], safe_flow_gt
            )
        else:
            i_loss = (pred - safe_flow_gt).abs()

        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, pred.shape]

        raw_den = valid_float.sum().clamp_min(1.0)
        raw_sequence_loss = raw_sequence_loss + i_weight * (i_loss * valid_float).sum() / raw_den

        weighted_valid = valid_float * loss_weight.to(dtype=i_loss.dtype)
        weighted_den = weighted_valid.sum().clamp_min(1.0)
        weighted_sequence_loss = weighted_sequence_loss + i_weight * (i_loss * weighted_valid).sum() / weighted_den

    final_pred = _as_disp_4d(flow_preds[-1])
    if use_disp_grad_loss:
        grad_loss = disparity_gradient_loss(final_pred, safe_flow_gt, valid)
    else:
        grad_loss = final_pred.sum() * 0.0

    non_degrade_terms = []
    if use_non_degrade_loss and len(flow_preds) > 1:
        for previous, current in zip(flow_preds[:-1], flow_preds[1:]):
            previous = _as_disp_4d(previous)
            current = _as_disp_4d(current)
            previous_error = (previous - safe_flow_gt).abs().detach()
            current_error = (current - safe_flow_gt).abs()
            stable_mask = valid & (
                previous_error < float(non_degrade_good_px)
            )
            degrade_map = F.relu(
                current_error - previous_error - float(non_degrade_margin)
            )
            non_degrade_terms.append(
                (degrade_map * stable_mask.float()).sum()
                / stable_mask.float().sum().clamp_min(1.0)
            )
    non_degrade_loss = (
        torch.stack(non_degrade_terms).mean()
        if non_degrade_terms else final_pred.sum() * 0.0
    )

    total_loss = (
        weighted_sequence_loss
        + float(lambda_grad) * grad_loss
        + float(lambda_non_degrade) * non_degrade_loss
    )

    epe = torch.sum((final_pred - safe_flow_gt) ** 2, dim=1).sqrt()
    valid_flat = valid.view(-1)
    epe = epe.view(-1)[valid_flat]

    if epe.numel() > 0:
        epe_mean = epe.mean().item()
        px1 = (epe < 1).float().mean().item()
        px3 = (epe < 3).float().mean().item()
        px5 = (epe < 5).float().mean().item()
    else:
        epe_mean = px1 = px3 = px5 = 0.0

    if delta_info_preds:
        confidence_values = [
            torch.softmax(info.float()[:, :2], dim=1)[:, 1:2].mean()
            for info in delta_info_preds
        ]
        log_b_values = [
            info.float()[:, 2:3].clamp(0.0, 10.0).mean()
            for info in delta_info_preds
        ]
        mol_confidence_mean = torch.stack(confidence_values).mean().detach().item()
        mol_log_b_mean = torch.stack(log_b_values).mean().detach().item()
    else:
        mol_confidence_mean = 0.0
        mol_log_b_mean = 0.0

    metrics = {
        'epe': epe_mean,
        '1px': px1,
        '3px': px3,
        '5px': px5,
        'sequence_loss_raw': raw_sequence_loss.detach().item(),
        'sequence_loss_edge_weighted': weighted_sequence_loss.detach().item(),
        'disp_grad_loss': grad_loss.detach().item(),
        'non_degrade_loss': non_degrade_loss.detach().item(),
        'edge_pixel_ratio': edge_ratio.detach().item(),
        'mol_confidence_mean': mol_confidence_mean,
        'mol_log_b_mean': mol_log_b_mean,
        'total_loss': total_loss.detach().item(),
    }

    return total_loss, metrics



def mixlap_loss(output, target, loss_gamma=0.9, max_disp=768):
    """Standalone MOL sequence loss using the PACT three-channel contract."""
    disp_gt = _as_disp_4d(target['disp']).float()
    valid = _as_disp_4d(target['valid'])
    finite = torch.isfinite(disp_gt)
    safe_gt = torch.where(finite, disp_gt, torch.zeros_like(disp_gt))
    valid = (valid >= 0.5) & finite & (safe_gt >= 0.0) & (safe_gt < max_disp)
    disp_preds = output['delta_disp_preds']
    info_preds = output['delta_info_preds']
    if len(disp_preds) != len(info_preds):
        raise ValueError("delta disparity/info prediction counts must match")

    lap_loss = safe_gt.sum() * 0.0
    n_predictions = len(disp_preds)
    for i in range(n_predictions):
        i_weight = loss_gamma**(n_predictions - i - 1)
        lap_term = _mixlap_nll(disp_preds[i], info_preds[i], safe_gt)
        weights = valid.to(lap_term.dtype)
        lap_loss = lap_loss + i_weight * (
            lap_term * weights
        ).sum() / weights.sum().clamp_min(1.0)

    return lap_loss

def _legacy_pact_auxiliary_loss(
    aux,
    disp_gt,
    valid,
    max_disp=None,
    coarse_cls_weight=0.5,
    coarse_reg_weight=0.2,
    tube_cls_weight=0.3,
    tube_reg_weight=0.1,
    radius_weight=0.05,
    base_weight=0.05,
    coarse_sigma=16.0,
    tube_sigma=4.0,
    smooth_l1_beta=1.0,
    radius_margin=4.0,
    radius_min=4.0,
    radius_max=64.0,
    radius_log_space=True,
):
    """PACT auxiliary supervision for the compressed volume and PRU.

    Every disparity-valued tensor in aux is in full-resolution pixels,
    irrespective of its spatial resolution. Supported optional keys are:

    - coarse_logits [B,D,H/16,W/16], coarse_candidates [D] or
      [1,D,1,1], coarse_valid [B,D,H/16,W/16], and coarse_disp.
    - tube_logits [B,K,H/8,W/8], tube_candidates [B,K,H/8,W/8],
      and tube_disp [B,1,H/8,W/8].
    - lists radius_predictions [B,1,H/4,W/4], base_logits,
      base_candidates, and base_valid [B,3,H/4,W/4].

    A regular coarse grid may omit coarse_candidates; its full-resolution
    candidates are inferred as arange(D) * aux.get('coarse_stride', 16).
    Missing branches contribute zero. Classification uses Gaussian soft labels
    over the actual candidate disparities. Radius targets are the nearest base
    candidate error plus 4 px, clamped to [4,64] px by default.

    Returns:
        A weighted scalar auxiliary loss and a detached metrics dictionary.
    """
    if aux is None:
        aux = {}
    if not isinstance(aux, dict):
        raise TypeError(f"PACT aux must be a dict or None, got {type(aux).__name__}")

    disp_gt = _as_disp_4d(disp_gt).float()
    valid = _as_disp_4d(valid)
    zero = torch.nan_to_num(disp_gt).sum() * 0.0

    def as_sequence(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if torch.is_tensor(value) and value.ndim == 5:
            return list(value.unbind(0))
        return [value]

    def at(sequence, index):
        if not sequence:
            return None
        if len(sequence) == 1:
            return sequence[0]
        return sequence[index] if index < len(sequence) else None

    def supervision(size):
        finite = torch.isfinite(disp_gt)
        mask = (valid >= 0.5) & finite & (disp_gt >= 0.0)
        if max_disp is not None:
            mask = mask & (disp_gt < float(max_disp))
        target = torch.where(finite, disp_gt, torch.zeros_like(disp_gt))
        if tuple(target.shape[-2:]) != tuple(size):
            target = F.interpolate(target, size=size, mode='nearest')
            mask = F.interpolate(mask.float(), size=size, mode='nearest') >= 0.5
        return target, mask

    def expand_candidates(candidates, reference, expected_count=None):
        if candidates is None:
            return None
        batch, _, height, width = reference.shape
        if not torch.is_tensor(candidates):
            candidates = torch.as_tensor(candidates, device=reference.device)
        candidates = candidates.to(device=reference.device, dtype=torch.float32)
        if candidates.ndim == 1:
            candidates = candidates[None, :, None, None]
        elif candidates.ndim == 2:
            candidates = candidates[:, :, None, None]
        elif candidates.ndim == 3:
            candidates = candidates[None]
        elif candidates.ndim != 4:
            raise ValueError(
                "PACT candidates require [K], [B,K], [K,H,W], or [B,K,H,W], "
                f"got {tuple(candidates.shape)}"
            )
        if expected_count is not None and candidates.shape[1] != expected_count:
            raise ValueError(
                f"PACT has {candidates.shape[1]} candidates but expects {expected_count}"
            )
        if candidates.shape[0] not in (1, batch):
            raise ValueError(
                f"PACT candidate batch {candidates.shape[0]} does not match {batch}"
            )
        if tuple(candidates.shape[-2:]) == (1, 1):
            candidates = candidates.expand(
                candidates.shape[0], candidates.shape[1], height, width
            )
        elif tuple(candidates.shape[-2:]) != (height, width):
            candidates = F.interpolate(
                candidates, size=(height, width), mode='bilinear',
                align_corners=True,
            )
        if candidates.shape[0] == 1 and batch > 1:
            candidates = candidates.expand(
                batch, candidates.shape[1], height, width
            )
        return candidates

    def masked_mean(values, mask, reference):
        weight = mask.to(dtype=values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1.0) + (
            torch.nan_to_num(reference.float()).sum() * 0.0
        )

    def expand_valid_mask(mask, reference, expected_count):
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=reference.device)
        mask = mask.to(device=reference.device)
        if mask.ndim == 3:
            mask = mask[None]
        if mask.ndim != 4:
            raise ValueError(
                f"PACT candidate-valid mask must be [B,K,H,W], got {tuple(mask.shape)}"
            )
        batch, _, height, width = reference.shape
        if mask.shape[1] != expected_count:
            raise ValueError(
                f"PACT valid mask has {mask.shape[1]} candidates but expects {expected_count}"
            )
        if mask.shape[0] not in (1, batch):
            raise ValueError(
                f"PACT valid-mask batch {mask.shape[0]} does not match {batch}"
            )
        if tuple(mask.shape[-2:]) != (height, width):
            mask = F.interpolate(
                mask.float(), size=(height, width), mode='nearest'
            )
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, expected_count, height, width)
        return mask >= 0.5

    def classify(logits, candidates, sigma, explicit_valid=None):
        if logits is None or candidates is None:
            return zero, None, None
        if not torch.is_tensor(logits) or logits.ndim != 4:
            raise ValueError("PACT logits must use [B,K,H,W]")
        candidates = expand_candidates(
            candidates, logits, expected_count=logits.shape[1]
        )
        target, target_valid = supervision(logits.shape[-2:])
        label_candidates = candidates.detach()
        candidate_valid = (
            torch.isfinite(label_candidates) & (label_candidates >= 0.0)
        )
        if max_disp is not None:
            candidate_valid = candidate_valid & (
                label_candidates < float(max_disp)
            )
        explicit_valid = expand_valid_mask(
            explicit_valid, logits, logits.shape[1]
        )
        if explicit_valid is not None:
            candidate_valid = candidate_valid & explicit_valid
        pixel_valid = target_valid[:, 0] & candidate_valid.any(dim=1)

        logits_fp32 = torch.nan_to_num(
            logits.float(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4
        )
        log_probability = F.log_softmax(
            logits_fp32.masked_fill(~candidate_valid, -1.0e9), dim=1
        )
        safe_labels = torch.where(
            candidate_valid, label_candidates, torch.zeros_like(label_candidates)
        )
        sigma = max(float(sigma), 1.0e-6)
        distance = ((safe_labels - target).abs() / sigma).clamp_max(1.0e4)
        energy = (-0.5 * distance.square()).masked_fill(
            ~candidate_valid, -1.0e9
        )
        energy = energy - energy.max(dim=1, keepdim=True).values
        soft_target = energy.exp() * candidate_valid.float()
        soft_target = soft_target / soft_target.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        loss_map = -(soft_target * log_probability).sum(dim=1)
        cls_loss = masked_mean(loss_map, pixel_valid, logits)

        safe_candidates = torch.where(
            torch.isfinite(candidates), candidates, torch.zeros_like(candidates)
        )
        posterior = (
            log_probability.exp() * safe_candidates
        ).sum(dim=1, keepdim=True)
        return cls_loss, posterior, pixel_valid[:, None]

    def regress(prediction, extra_valid=None):
        if prediction is None:
            return zero
        prediction = _as_disp_4d(prediction).float()
        if prediction.shape[1] != 1:
            raise ValueError(
                "PACT disparity regression requires [B,1,H,W], got "
                f"{tuple(prediction.shape)}"
            )
        target, target_valid = supervision(prediction.shape[-2:])
        if extra_valid is not None:
            if tuple(extra_valid.shape[-2:]) != tuple(prediction.shape[-2:]):
                extra_valid = F.interpolate(
                    extra_valid.float(), size=prediction.shape[-2:],
                    mode='nearest',
                ) >= 0.5
            target_valid = target_valid & extra_valid.bool()
        target_valid = target_valid & torch.isfinite(prediction)
        safe_prediction = torch.nan_to_num(
            prediction, nan=0.0, posinf=1.0e4, neginf=-1.0e4
        )
        loss_map = F.smooth_l1_loss(
            safe_prediction, target, reduction='none',
            beta=float(smooth_l1_beta),
        )
        return masked_mean(loss_map, target_valid, prediction)

    coarse_logits = aux.get('coarse_logits')
    coarse_candidates = aux.get('coarse_candidates')
    if coarse_logits is not None and coarse_candidates is None:
        if not torch.is_tensor(coarse_logits) or coarse_logits.ndim != 4:
            raise ValueError("coarse_logits must use [B,D,H,W]")
        coarse_candidates = torch.arange(
            coarse_logits.shape[1], device=coarse_logits.device,
            dtype=torch.float32,
        ) * float(aux.get('coarse_stride', 16.0))
    coarse_cls, coarse_posterior, coarse_mask = classify(
        coarse_logits, coarse_candidates, coarse_sigma,
        aux.get('coarse_valid'),
    )
    coarse_direct = aux.get('coarse_disp')
    coarse_reg = regress(
        coarse_direct if coarse_direct is not None else coarse_posterior,
        coarse_mask,
    )

    tube_logits = aux.get('tube_logits')
    tube_candidates = aux.get('tube_candidates')
    tube_cls, tube_posterior, tube_mask = classify(
        tube_logits, tube_candidates, tube_sigma,
        aux.get('tube_valid'),
    )
    tube_direct = aux.get('tube_disp')
    tube_reg = regress(
        tube_direct if tube_direct is not None else tube_posterior,
        tube_mask,
    )

    base_logits = as_sequence(aux.get('base_logits'))
    base_candidates = as_sequence(aux.get('base_candidates'))
    base_valid = as_sequence(aux.get('base_valid'))
    radius_predictions = as_sequence(aux.get('radius_predictions'))
    rounds = max(
        len(base_logits), len(base_candidates), len(base_valid),
        len(radius_predictions),
    )
    base_losses = []
    radius_losses = []
    for index in range(rounds):
        logits_i = at(base_logits, index)
        candidates_i = at(base_candidates, index)
        explicit_valid_i = at(base_valid, index)
        radius_i = at(radius_predictions, index)
        if candidates_i is None or (logits_i is None and radius_i is None):
            continue
        reference = logits_i if logits_i is not None else _as_disp_4d(radius_i)
        if logits_i is not None:
            if not torch.is_tensor(logits_i) or logits_i.ndim != 4:
                raise ValueError("Each base_logits item requires [B,K,H,W]")
            expected_count = logits_i.shape[1]
        else:
            expected_count = None
        candidates_i = expand_candidates(
            candidates_i, reference, expected_count=expected_count
        )
        target, target_valid = supervision(reference.shape[-2:])
        candidate_valid = (
            torch.isfinite(candidates_i.detach())
            & (candidates_i.detach() >= 0.0)
        )
        if max_disp is not None:
            candidate_valid = candidate_valid & (
                candidates_i.detach() < float(max_disp)
            )
        explicit_valid_i = expand_valid_mask(
            explicit_valid_i, reference, candidates_i.shape[1]
        )
        if explicit_valid_i is not None:
            candidate_valid = candidate_valid & explicit_valid_i
        safe_candidates = torch.where(
            candidate_valid, candidates_i.detach(),
            torch.zeros_like(candidates_i.detach()),
        )
        candidate_error = (safe_candidates - target).abs().masked_fill(
            ~candidate_valid, 1.0e9
        )
        nearest_error, nearest_index = candidate_error.min(dim=1)
        round_valid = target_valid[:, 0] & candidate_valid.any(dim=1)

        if logits_i is not None:
            safe_logits = torch.nan_to_num(
                logits_i.float(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4
            ).masked_fill(~candidate_valid, -1.0e9)
            base_map = F.cross_entropy(
                safe_logits, nearest_index, reduction='none'
            )
            base_losses.append(
                masked_mean(base_map, round_valid, logits_i)
            )

        if radius_i is not None:
            radius_i = _as_disp_4d(radius_i).float()
            if radius_i.shape[1] != 1:
                raise ValueError(
                    "Each radius_predictions item requires [B,1,H,W]"
                )
            radius_target = (
                nearest_error[:, None] + float(radius_margin)
            ).clamp(float(radius_min), float(radius_max)).detach()
            radius_valid = round_valid[:, None]
            if tuple(radius_i.shape[-2:]) != tuple(reference.shape[-2:]):
                radius_target = F.interpolate(
                    radius_target, size=radius_i.shape[-2:], mode='nearest'
                )
                radius_valid = F.interpolate(
                    radius_valid.float(), size=radius_i.shape[-2:],
                    mode='nearest',
                ) >= 0.5
            radius_valid = radius_valid & torch.isfinite(radius_i)
            safe_radius = torch.nan_to_num(
                radius_i, nan=float(radius_min), posinf=float(radius_max),
                neginf=float(radius_min),
            ).clamp_min(1.0e-6)
            if radius_log_space:
                radius_map = F.smooth_l1_loss(
                    safe_radius.log(), radius_target.clamp_min(1.0e-6).log(),
                    reduction='none', beta=float(smooth_l1_beta),
                )
            else:
                radius_map = F.smooth_l1_loss(
                    safe_radius, radius_target, reduction='none',
                    beta=float(smooth_l1_beta),
                )
            radius_losses.append(
                masked_mean(radius_map, radius_valid, radius_i)
            )

    base_loss = torch.stack(base_losses).mean() if base_losses else zero
    radius_loss = torch.stack(radius_losses).mean() if radius_losses else zero
    total_loss = (
        float(coarse_cls_weight) * coarse_cls
        + float(coarse_reg_weight) * coarse_reg
        + float(tube_cls_weight) * tube_cls
        + float(tube_reg_weight) * tube_reg
        + float(radius_weight) * radius_loss
        + float(base_weight) * base_loss
    )
    metrics = {
        'pact_aux_loss': total_loss.detach().item(),
        'pact_coarse_cls_loss': coarse_cls.detach().item(),
        'pact_coarse_reg_loss': coarse_reg.detach().item(),
        'pact_tube_cls_loss': tube_cls.detach().item(),
        'pact_tube_reg_loss': tube_reg.detach().item(),
        'pact_radius_loss': radius_loss.detach().item(),
        'pact_base_loss': base_loss.detach().item(),
        'pact_base_supervised_rounds': float(len(base_losses)),
        'pact_radius_supervised_rounds': float(len(radius_losses)),
    }
    return total_loss, metrics


def pact_auxiliary_loss(aux, disp_gt, valid, max_disp=None,
                        coarse_cls_weight=0.5, coarse_reg_weight=0.2, init_weight=0.3,
                        mono_reg_weight=0.05, confidence_weight=0.05,
                        coarse_sigma=16.0, smooth_l1_beta=1.0):
    """Auxiliary objective for the single-current-disparity PACT model.

    No anchor, tube or base-selection tensors are accepted by the active
    objective.  All disparity-valued entries in ``aux`` use full-resolution
    pixel units even when their spatial grid is lower resolution.
    """
    if not isinstance(aux, dict):
        raise TypeError(f"PACT aux must be a dict, got {type(aux).__name__}")

    disp_gt = _as_disp_4d(disp_gt).float()
    valid = _as_disp_4d(valid)
    finite_gt = torch.isfinite(disp_gt)
    safe_gt = torch.where(finite_gt, disp_gt, torch.zeros_like(disp_gt))
    supervision_valid = (valid >= 0.5) & finite_gt & (safe_gt >= 0.0)
    if max_disp is not None:
        supervision_valid = supervision_valid & (safe_gt < float(max_disp))
    zero = safe_gt.sum() * 0.0

    def supervision(size):
        target = safe_gt
        mask = supervision_valid
        if tuple(target.shape[-2:]) != tuple(size):
            target = F.interpolate(target, size=size, mode="nearest")
            mask = F.interpolate(mask.float(), size=size, mode="nearest") >= 0.5
        return target, mask

    def masked_mean(values, mask, reference):
        weights = mask.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0) + (
            torch.nan_to_num(reference.float()).sum() * 0.0
        )

    def regress(prediction):
        if prediction is None:
            return zero
        prediction = _as_disp_4d(prediction).float()
        if prediction.shape[1] != 1:
            raise ValueError(
                f"PACT regression expects [B,1,H,W], got {tuple(prediction.shape)}"
            )
        target, mask = supervision(prediction.shape[-2:])
        mask = mask & torch.isfinite(prediction)
        safe_prediction = torch.nan_to_num(
            prediction, nan=0.0, posinf=1.0e4, neginf=-1.0e4
        )
        loss_map = F.smooth_l1_loss(
            safe_prediction,
            target,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
        return masked_mean(loss_map, mask, prediction)

    coarse_logits = aux.get("coarse_logits")
    coarse_candidates = aux.get("coarse_candidates")
    coarse_valid = aux.get("coarse_valid")
    if not torch.is_tensor(coarse_logits) or coarse_logits.ndim != 4:
        raise ValueError("coarse_logits must be [B,D,H,W]")
    batch, bins, height, width = coarse_logits.shape
    if coarse_candidates is None:
        coarse_candidates = torch.arange(
            bins, device=coarse_logits.device, dtype=torch.float32
        ).view(1, bins, 1, 1) * 16.0
    else:
        coarse_candidates = torch.as_tensor(
            coarse_candidates, device=coarse_logits.device, dtype=torch.float32
        )
        if coarse_candidates.ndim == 1:
            coarse_candidates = coarse_candidates.view(1, bins, 1, 1)
    if coarse_candidates.shape[1] != bins:
        raise ValueError("coarse candidate count does not match coarse logits")
    coarse_candidates = coarse_candidates.expand(batch, bins, height, width)
    if coarse_valid is None:
        coarse_valid = torch.ones_like(coarse_logits, dtype=torch.bool)
    else:
        coarse_valid = coarse_valid.bool()
        if coarse_valid.shape != coarse_logits.shape:
            raise ValueError("coarse_valid must match coarse_logits")
    coarse_valid = coarse_valid & torch.isfinite(coarse_candidates)
    if max_disp is not None:
        coarse_valid = coarse_valid & (coarse_candidates < float(max_disp))

    coarse_target, coarse_pixel_valid = supervision((height, width))
    coarse_pixel_valid = coarse_pixel_valid[:, 0] & coarse_valid.any(dim=1)
    safe_logits = torch.nan_to_num(
        coarse_logits.float(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4
    ).masked_fill(~coarse_valid, -1.0e9)
    log_probability = F.log_softmax(safe_logits, dim=1)
    distance = (
        (coarse_candidates - coarse_target).abs()
        / max(float(coarse_sigma), 1.0e-6)
    ).clamp_max(1.0e4)
    energy = (-0.5 * distance.square()).masked_fill(~coarse_valid, -1.0e9)
    energy = energy - energy.max(dim=1, keepdim=True).values
    soft_target = energy.exp() * coarse_valid.float()
    soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    coarse_cls_map = -(soft_target * log_probability).sum(dim=1)
    coarse_cls = masked_mean(
        coarse_cls_map, coarse_pixel_valid, coarse_logits
    )
    coarse_reg = regress(aux.get("coarse_disp"))
    init_reg = regress(aux.get("init_disp"))
    mono_reg = regress(aux.get("mono_calibrated"))

    confidence = aux.get("mono_confidence")
    if confidence is None:
        confidence_loss = zero
    else:
        confidence = _as_disp_4d(confidence).float()
        target, mask = supervision(confidence.shape[-2:])
        coarse_for_conf = F.interpolate(
            _as_disp_4d(aux["coarse_disp"]).float(),
            size=confidence.shape[-2:], mode="bilinear", align_corners=True,
        )
        mono_for_conf = F.interpolate(
            _as_disp_4d(aux["mono_calibrated"]).float(),
            size=confidence.shape[-2:], mode="bilinear", align_corners=True,
        )
        confidence_target = torch.sigmoid(
            ((coarse_for_conf - target).abs()
             - (mono_for_conf - target).abs()) / 4.0
        ).detach()
        confidence_safe = torch.nan_to_num(confidence).clamp(1.0e-6, 1.0 - 1.0e-6)
        confidence_map = F.binary_cross_entropy(
            confidence_safe, confidence_target, reduction="none"
        )
        confidence_loss = masked_mean(
            confidence_map,
            mask & torch.isfinite(confidence),
            confidence,
        )

    total_loss = (
        float(coarse_cls_weight) * coarse_cls
        + float(coarse_reg_weight) * coarse_reg
        + float(init_weight) * init_reg
        + float(mono_reg_weight) * mono_reg
        + float(confidence_weight) * confidence_loss
    )
    metrics = {
        "pact_aux_loss": total_loss.detach().item(),
        "pact_coarse_cls_loss": coarse_cls.detach().item(),
        "pact_coarse_reg_loss": coarse_reg.detach().item(),
        "pact_init_loss": init_reg.detach().item(),
        "pact_mono_reg_loss": mono_reg.detach().item(),
        "pact_confidence_loss": confidence_loss.detach().item(),
    }
    return total_loss, metrics
