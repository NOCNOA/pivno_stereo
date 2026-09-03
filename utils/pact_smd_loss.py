"""Losses for the isolated PACT-SMD d0 experiment."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.utils import disparity_edge_mask


def _as_disp_4d(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.unsqueeze(1)
    if value.ndim == 4:
        return value
    raise ValueError(f"expected [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0) + (
        torch.nan_to_num(reference.float()).sum() * 0.0
    )


def _full_resolution_samples(
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    output_size: Tuple[int, int],
    max_disp: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 4x4 GT samples and masks for every quarter-resolution cell."""

    disp_gt = _as_disp_4d(disp_gt).float()
    valid = _as_disp_4d(valid)
    batch, channels, height, width = disp_gt.shape
    if channels != 1:
        raise ValueError("PACT-SMD disparity GT must have one channel")
    quarter_height, quarter_width = map(int, output_size)
    if (height, width) != (quarter_height * 4, quarter_width * 4):
        raise ValueError(
            "PACT-SMD 4x4 supervision requires an exact quarter grid: "
            f"gt={(height, width)}, mixture={output_size}"
        )

    finite = torch.isfinite(disp_gt)
    safe_gt = torch.where(finite, disp_gt, torch.zeros_like(disp_gt))
    valid_full = (valid >= 0.5) & finite & (safe_gt >= 0.0)
    if max_disp is not None:
        valid_full = valid_full & (safe_gt < float(max_disp))

    samples = F.unfold(safe_gt, kernel_size=4, stride=4).view(
        batch, 16, quarter_height, quarter_width
    )
    sample_valid = F.unfold(
        valid_full.float(), kernel_size=4, stride=4
    ).view(batch, 16, quarter_height, quarter_width) >= 0.5

    edge_full, _, _ = disparity_edge_mask(
        safe_gt,
        valid_full,
        edge_mode="threshold",
        edge_threshold=1.0,
        edge_dilation=5,
    )
    edge_cell = F.max_pool2d(edge_full, kernel_size=4, stride=4) > 0.0
    return samples, sample_valid, edge_cell, safe_gt


def bimodal_laplace_nll(
    means: torch.Tensor,
    scales: torch.Tensor,
    mixture_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-target negative log likelihood for a two-component Laplace mix.

    Args:
        means/scales/logits: ``[B,2,H,W]``.
        targets: ``[B,N,H,W]``.
    Returns:
        NLL map ``[B,N,H,W]``.
    """

    expected = means.shape
    if means.ndim != 4 or means.shape[1] != 2:
        raise ValueError(f"means must be [B,2,H,W], got {tuple(means.shape)}")
    if scales.shape != expected or mixture_logits.shape != expected:
        raise ValueError("PACT-SMD means, scales and logits must have equal shapes")
    if targets.ndim != 4 or targets.shape[0] != means.shape[0] or targets.shape[-2:] != means.shape[-2:]:
        raise ValueError("PACT-SMD target samples do not align with mixture maps")
    safe_scales = torch.nan_to_num(
        scales.float(), nan=1.0, posinf=64.0, neginf=1.0
    ).clamp_min(1.0e-4)
    log_weights = F.log_softmax(
        torch.nan_to_num(mixture_logits.float()), dim=1
    )
    component_log_probability = (
        log_weights.unsqueeze(2)
        - math.log(2.0)
        - torch.log(safe_scales).unsqueeze(2)
        - (targets.float().unsqueeze(1) - means.float().unsqueeze(2)).abs()
        / safe_scales.unsqueeze(2)
    )
    return -torch.logsumexp(component_log_probability, dim=1)


def pact_smd_auxiliary_loss(
    auxiliary: Dict[str, object],
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_disp: Optional[float],
    stage: str,
    nll_weight: float = 1.0,
    selection_weight: float = 0.2,
    guard_weight: float = 0.05,
    nll_edge_only: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train the SMD density, mode selector and safe d0 fallback."""

    if stage not in ("head", "joint", "full"):
        raise ValueError(
            "PACT-SMD loss stage must be 'head', 'joint' or 'full'"
        )
    required = (
        "smd_means",
        "smd_scales",
        "smd_mixture_logits",
        "smd_peak_scores",
        "smd_selected_mode",
        "init_disp",
        "init_disp_old",
    )
    missing = [name for name in required if not torch.is_tensor(auxiliary.get(name))]
    if missing:
        raise ValueError(f"PACT-SMD auxiliary output is missing tensors: {missing}")

    means = auxiliary["smd_means"].float()
    scales = auxiliary["smd_scales"].float()
    mixture_logits = auxiliary["smd_mixture_logits"].float()
    peak_scores = auxiliary["smd_peak_scores"].float()
    selected_mode = auxiliary["smd_selected_mode"].float()
    d0_new = auxiliary["init_disp"].float()
    d0_old = auxiliary["init_disp_old"].float()
    samples, sample_valid, edge_cell, safe_gt = _full_resolution_samples(
        disp_gt, valid, means.shape[-2:], max_disp
    )

    nll_map = bimodal_laplace_nll(means, scales, mixture_logits, samples)
    cell_weight = 1.0 + 2.0 * edge_cell.float()
    weighted_sample_valid = sample_valid.float() * cell_weight
    if nll_edge_only:
        weighted_sample_valid = weighted_sample_valid * edge_cell.float()
    nll_loss = _masked_mean(
        nll_map, weighted_sample_valid, mixture_logits
    )

    absolute_error = (
        means.unsqueeze(2) - samples.unsqueeze(1)
    ).abs()
    valid_per_mode = sample_valid.unsqueeze(1).to(absolute_error.dtype)
    mode_error = (
        absolute_error * valid_per_mode
    ).sum(dim=2) / valid_per_mode.sum(dim=2).clamp_min(1.0)
    winner = mode_error.argmin(dim=1)
    cell_valid = sample_valid.any(dim=1)
    selection_map = F.cross_entropy(
        peak_scores,
        winner,
        reduction="none",
    )
    selection_loss = _masked_mean(
        selection_map,
        cell_valid & torch.isfinite(selection_map),
        peak_scores,
    )

    target_nearest = F.interpolate(
        safe_gt, size=d0_new.shape[-2:], mode="nearest"
    )
    raw_gt = _as_disp_4d(disp_gt)
    valid_full = (
        (_as_disp_4d(valid) >= 0.5)
        & torch.isfinite(raw_gt)
        & (raw_gt >= 0.0)
    )
    valid_nearest = F.interpolate(
        valid_full.float(), size=d0_new.shape[-2:], mode="nearest"
    ) >= 0.5
    if max_disp is not None:
        valid_nearest = valid_nearest & (target_nearest < float(max_disp))
    old_error = (d0_old - target_nearest).abs()
    new_error = (d0_new - target_nearest).abs()
    old_good = old_error < 3.0
    guard_map = F.relu(new_error - old_error - 0.5)
    guard_loss = _masked_mean(
        guard_map,
        valid_nearest & old_good & torch.isfinite(guard_map),
        d0_new,
    )

    total = (
        float(nll_weight) * nll_loss
        + float(selection_weight) * selection_loss
        + float(guard_weight) * guard_loss
    )
    edge = edge_cell & cell_valid.unsqueeze(1)
    interior = (~edge_cell) & cell_valid.unsqueeze(1)
    mode2_edge = _masked_mean(selected_mode, edge, selected_mode)
    mode2_interior = _masked_mean(selected_mode, interior, selected_mode)
    separation = (means[:, 0:1] - means[:, 1:2]).abs()
    metrics = {
        "smd_aux_loss": total.detach().item(),
        "smd_nll_loss": nll_loss.detach().item(),
        "smd_nll_edge_only": float(bool(nll_edge_only)),
        "smd_selection_loss": selection_loss.detach().item(),
        "smd_guard_loss": guard_loss.detach().item(),
        "smd_mode2_edge_ratio": mode2_edge.detach().item(),
        "smd_mode2_interior_ratio": mode2_interior.detach().item(),
        "smd_mode_separation_px": separation.mean().detach().item(),
        "smd_scale1_px": scales[:, 0:1].mean().detach().item(),
        "smd_scale2_px": scales[:, 1:2].mean().detach().item(),
    }
    return total, metrics


__all__ = ["bimodal_laplace_nll", "pact_smd_auxiliary_loss"]
