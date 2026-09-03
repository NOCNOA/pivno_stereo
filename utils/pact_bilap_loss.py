"""Sequence objective for recurrent bimodal Laplace disparity predictions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.utils import disparity_edge_mask


def _as_4d(value):
    return value.unsqueeze(1) if value.ndim == 3 else value


def _masked_mean(value, weight):
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def mixture_laplace_nll(means, scales, logits, target):
    scales = torch.nan_to_num(scales.float(), nan=1.0, posinf=64.0, neginf=1.0).clamp_min(1.0e-4)
    log_weights = F.log_softmax(torch.nan_to_num(logits.float()), dim=1)
    component_log_probability = log_weights - math.log(2.0) - torch.log(scales) - (target.float() - means.float()).abs() / scales
    return -torch.logsumexp(component_log_probability, dim=1, keepdim=True)


def bilap_sequence_loss(predictions, disp_gt, valid, max_disp=None, gamma=0.9, nll_weight=1.0, map_weight=1.0, edge_weight=2.0, diversity_weight=0.0, diversity_margin=3.0, nll_edge_only=True):
    """Supervise every recurrent distribution at full image resolution."""
    if not predictions:
        raise ValueError("BiLap sequence loss requires at least one prediction")
    target = _as_4d(disp_gt).float()
    valid = _as_4d(valid) >= 0.5
    finite = torch.isfinite(target)
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    valid = valid & finite & (safe_target >= 0.0)
    if max_disp is not None:
        valid = valid & (safe_target < float(max_disp))
    edge, _, _ = disparity_edge_mask(safe_target, valid, edge_mode="threshold", edge_threshold=1.0, edge_dilation=5)
    pixel_weight = valid.float() * (1.0 + float(edge_weight) * edge.float())
    nll_pixel_weight = (valid & edge.bool()).float() if nll_edge_only else pixel_weight
    total = safe_target.new_zeros(())
    last_nll = last_map = last_diversity = safe_target.new_zeros(())
    count = len(predictions)
    for index, prediction in enumerate(predictions):
        means = prediction["means"].float()
        scales = prediction["scales"].float()
        logits = prediction["mixture_logits"].float()
        point = prediction["disp"].float()
        nll = mixture_laplace_nll(means, scales, logits, safe_target)
        map_error = F.smooth_l1_loss(point, safe_target, reduction="none", beta=1.0)
        nll_loss = _masked_mean(nll, nll_pixel_weight)
        map_loss = _masked_mean(map_error, pixel_weight)
        separation = (means[:, 0:1] - means[:, -1:]).abs()
        balance = torch.softmax(logits.float(), dim=1).prod(dim=1, keepdim=True)
        diversity = _masked_mean(balance * F.relu(float(diversity_margin) - separation), valid & edge.bool()) if means.shape[1] > 1 else separation.sum() * 0.0
        iteration_weight = float(gamma) ** (count - index - 1)
        total = total + iteration_weight * (float(nll_weight) * nll_loss + float(map_weight) * map_loss + float(diversity_weight) * diversity)
        last_nll, last_map, last_diversity = nll_loss, map_loss, diversity
    final = predictions[-1]
    absolute_error = (final["disp"].float() - safe_target).abs()
    valid_float = valid.float()
    epe = _masked_mean(absolute_error, valid_float)
    out3 = _masked_mean((absolute_error > 3.0).float(), valid_float)
    separation = final["separation"].float()
    collapse = _masked_mean((separation < float(diversity_margin)).float(), valid_float)
    metrics = {"bilap_sequence_loss": total.detach().item(), "bilap_nll": last_nll.detach().item(), "bilap_nll_edge_only": float(bool(nll_edge_only)), "bilap_nll_pixel_ratio": (nll_pixel_weight.sum() / valid_float.sum().clamp_min(1.0)).detach().item(), "bilap_map_smooth_l1": last_map.detach().item(), "bilap_diversity": last_diversity.detach().item(), "bilap_final_epe": epe.detach().item(), "bilap_final_out3": out3.detach().item(), "bilap_mode_separation_px": _masked_mean(separation, valid_float).detach().item(), "bilap_collapse_ratio": collapse.detach().item(), "bilap_entropy": _masked_mean(final["entropy"].float(), valid_float).detach().item()}
    return total, metrics


__all__ = ["bilap_sequence_loss", "mixture_laplace_nll"]
