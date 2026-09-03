"""Loss terms for the frozen post-GRU bimodal Laplace experiment."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.pact_smd_loss import (
    _as_disp_4d,
    _full_resolution_samples,
    _masked_mean,
    bimodal_laplace_nll,
)
from utils.utils import disparity_edge_mask


def pact_smd_post_auxiliary_loss(
    auxiliary: Dict[str, object],
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_disp: Optional[float],
    nll_weight: float = 1.0,
    selection_weight: float = 0.2,
    guard_weight: float = 0.2,
    bad3_weight: float = 0.05,
    edge_weight: float = 2.0,
    guard_margin: float = 0.25,
    bad3_tau: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervise density/selection and prevent damage to good frozen pixels.

    The ordinary sequence loss supplies the differentiable MAP disparity term.
    This function adds mixture NLL, winner classification, conservative
    degradation protection and a smooth Bad3 surrogate.
    """

    required = (
        "post_smd_means",
        "post_smd_scales",
        "post_smd_mixture_logits",
        "post_smd_peak_scores",
        "post_smd_selected_mode",
        "post_raw_final",
        "post_final",
    )
    missing = [name for name in required if not torch.is_tensor(auxiliary.get(name))]
    if missing:
        raise ValueError(f"post-GRU SMD auxiliary output is missing: {missing}")
    if bad3_tau <= 0.0:
        raise ValueError("post-GRU Bad3 temperature must be positive")

    means = auxiliary["post_smd_means"].float()
    scales = auxiliary["post_smd_scales"].float()
    logits = auxiliary["post_smd_mixture_logits"].float()
    peak_scores = auxiliary["post_smd_peak_scores"].float()
    selected_mode = auxiliary["post_smd_selected_mode"].float()
    raw_final = auxiliary["post_raw_final"].float()
    post_final = auxiliary["post_final"].float()

    samples, sample_valid, edge_cell, safe_gt = _full_resolution_samples(
        disp_gt, valid, means.shape[-2:], max_disp
    )
    nll_map = bimodal_laplace_nll(means, scales, logits, samples)
    cell_weight = 1.0 + float(edge_weight) * edge_cell.float()
    nll_loss = _masked_mean(
        nll_map,
        sample_valid.float() * cell_weight,
        logits,
    )

    per_mode_error = (means.unsqueeze(2) - samples.unsqueeze(1)).abs()
    mode_valid = sample_valid.unsqueeze(1).to(per_mode_error.dtype)
    per_mode_error = (
        (per_mode_error * mode_valid).sum(dim=2)
        / mode_valid.sum(dim=2).clamp_min(1.0)
    )
    winner = per_mode_error.argmin(dim=1)
    cell_valid = sample_valid.any(dim=1)
    selection_map = F.cross_entropy(peak_scores, winner, reduction="none")
    selection_loss = _masked_mean(
        selection_map,
        cell_valid.float()
        * (1.0 + float(edge_weight) * edge_cell[:, 0].float()),
        peak_scores,
    )

    target = _as_disp_4d(disp_gt).float()
    valid_full = (
        (_as_disp_4d(valid) >= 0.5)
        & torch.isfinite(target)
        & (target >= 0.0)
    )
    if max_disp is not None:
        valid_full = valid_full & (target < float(max_disp))
    edge_full, _, _ = disparity_edge_mask(
        safe_gt,
        valid_full,
        edge_mode="threshold",
        edge_threshold=1.0,
        edge_dilation=5,
    )
    pixel_weight = 1.0 + float(edge_weight) * edge_full.float()
    raw_error = (raw_final - safe_gt).abs()
    post_error = (post_final - safe_gt).abs()

    raw_good = raw_error < 3.0
    guard_map = F.relu(post_error - raw_error - float(guard_margin))
    guard_loss = _masked_mean(
        guard_map,
        valid_full.float() * raw_good.float() * pixel_weight,
        post_final,
    )
    bad3_map = torch.sigmoid((post_error - 3.0) / float(bad3_tau))
    bad3_loss = _masked_mean(
        bad3_map,
        valid_full.float() * pixel_weight,
        post_final,
    )

    total = (
        float(nll_weight) * nll_loss
        + float(selection_weight) * selection_loss
        + float(guard_weight) * guard_loss
        + float(bad3_weight) * bad3_loss
    )
    edge_valid = valid_full & (edge_full > 0.0)
    interior_valid = valid_full & ~(edge_full > 0.0)
    separation = (means[:, 0:1] - means[:, 1:2]).abs()
    metrics = {
        "post_smd_aux_loss": total.detach().item(),
        "post_smd_nll_loss": nll_loss.detach().item(),
        "post_smd_selection_loss": selection_loss.detach().item(),
        "post_smd_guard_loss": guard_loss.detach().item(),
        "post_smd_bad3_surrogate": bad3_loss.detach().item(),
        "post_smd_raw_epe": _masked_mean(
            raw_error, valid_full, raw_final
        ).detach().item(),
        "post_smd_final_epe": _masked_mean(
            post_error, valid_full, post_final
        ).detach().item(),
        "post_smd_edge_epe": _masked_mean(
            post_error, edge_valid, post_final
        ).detach().item(),
        "post_smd_interior_epe": _masked_mean(
            post_error, interior_valid, post_final
        ).detach().item(),
        "post_smd_mode2_ratio": selected_mode.mean().detach().item(),
        "post_smd_mode_separation_px": separation.mean().detach().item(),
        "post_smd_scale1_px": scales[:, 0:1].mean().detach().item(),
        "post_smd_scale2_px": scales[:, 1:2].mean().detach().item(),
    }
    return total, metrics


__all__ = ["pact_smd_post_auxiliary_loss"]
