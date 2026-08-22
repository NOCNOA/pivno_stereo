"""Stage-separated losses for the PACT-PIVNO model."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.utils import disparity_edge_mask, disparity_gradient_loss


def _as_disp_4d(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.unsqueeze(1)
    if value.ndim == 4:
        return value
    raise ValueError(
        f"expected disparity [B,H,W] or [B,1,H,W], got {tuple(value.shape)}"
    )


def _sequence_weight(index: int, count: int, gamma: float) -> float:
    adjusted_gamma = float(gamma) ** (15.0 / count)
    return adjusted_gamma ** (count - index - 1)


def _check_prediction(
    prediction: torch.Tensor,
    expected_shape: torch.Size,
    stage: str,
) -> torch.Tensor:
    prediction = _as_disp_4d(prediction)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"{stage} shape must be {tuple(expected_shape)}, "
            f"got {tuple(prediction.shape)}"
        )
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError(f"non-finite prediction in {stage}")
    return prediction


def pact_pivno_sequence_loss(
    init_predictions: Sequence[torch.Tensor],
    recurrent_predictions: Sequence[torch.Tensor],
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_disp: float = 768.0,
    gamma: float = 0.9,
    smooth_l1_beta: float = 1.0,
    edge_topk: float = 0.10,
    edge_dilation: int = 5,
    edge_weight: float = 2.0,
    gradient_weight: float = 0.05,
    non_degrade_weight: float = 0.02,
    non_degrade_good_px: float = 3.0,
    non_degrade_margin: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervise every PIVNO initialization and every recurrent refinement.

    Initialization predictions use Smooth L1 on ordinary valid pixels.
    Recurrent predictions use GT-edge-weighted L1. The final recurrent output
    retains the repository's gradient-consistency regularizer, and recurrent
    transitions retain the non-degradation guard.
    """
    if not init_predictions:
        raise ValueError("init_predictions must contain at least one prediction")
    if not recurrent_predictions:
        raise ValueError("recurrent_predictions must contain at least one prediction")

    disp_gt = _as_disp_4d(disp_gt).float()
    valid = _as_disp_4d(valid)
    finite_gt = torch.isfinite(disp_gt)
    safe_gt = torch.where(finite_gt, disp_gt, torch.zeros_like(disp_gt))
    valid_mask = (
        (valid >= 0.5)
        & finite_gt
        & (safe_gt.abs() < float(max_disp))
    )
    valid_float = valid_mask.float()
    valid_denominator = valid_float.sum().clamp_min(1.0)

    checked_init = [
        _check_prediction(prediction, safe_gt.shape, f"init[{index}]")
        for index, prediction in enumerate(init_predictions)
    ]
    checked_recurrent = [
        _check_prediction(prediction, safe_gt.shape, f"recurrent[{index}]")
        for index, prediction in enumerate(recurrent_predictions)
    ]

    init_loss = safe_gt.new_zeros(())
    for index, prediction in enumerate(checked_init):
        per_pixel = F.smooth_l1_loss(
            prediction.float(),
            safe_gt,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
        init_loss = init_loss + _sequence_weight(
            index, len(checked_init), gamma
        ) * (per_pixel * valid_float).sum() / valid_denominator

    edge_mask, edge_ratio, _ = disparity_edge_mask(
        safe_gt,
        valid_mask,
        edge_mode="topk",
        edge_topk=edge_topk,
        edge_dilation=edge_dilation,
    )
    recurrent_weights = valid_float * (
        1.0 + float(edge_weight) * edge_mask
    )
    recurrent_denominator = recurrent_weights.sum().clamp_min(1.0)

    recurrent_edge_loss = safe_gt.new_zeros(())
    recurrent_raw_loss = safe_gt.new_zeros(())
    for index, prediction in enumerate(checked_recurrent):
        per_pixel = (prediction.float() - safe_gt).abs()
        weight = _sequence_weight(index, len(checked_recurrent), gamma)
        recurrent_raw_loss = recurrent_raw_loss + weight * (
            per_pixel * valid_float
        ).sum() / valid_denominator
        recurrent_edge_loss = recurrent_edge_loss + weight * (
            per_pixel * recurrent_weights
        ).sum() / recurrent_denominator

    final_prediction = checked_recurrent[-1].float()
    gradient_loss = disparity_gradient_loss(
        final_prediction,
        safe_gt,
        valid_mask,
    )

    non_degrade_terms = []
    refinement_sequence = [checked_init[-1], *checked_recurrent]
    for previous, current in zip(
        refinement_sequence[:-1], refinement_sequence[1:]
    ):
        previous_error = (previous.float() - safe_gt).abs().detach()
        current_error = (current.float() - safe_gt).abs()
        stable_mask = valid_mask & (
            previous_error < float(non_degrade_good_px)
        )
        degradation = F.relu(
            current_error - previous_error - float(non_degrade_margin)
        )
        stable_float = stable_mask.float()
        non_degrade_terms.append(
            (degradation * stable_float).sum()
            / stable_float.sum().clamp_min(1.0)
        )
    non_degrade_loss = torch.stack(non_degrade_terms).mean()

    total_loss = (
        init_loss
        + recurrent_edge_loss
        + float(gradient_weight) * gradient_loss
        + float(non_degrade_weight) * non_degrade_loss
    )

    final_error = (final_prediction - safe_gt).abs()[valid_mask]
    if final_error.numel() > 0:
        epe = final_error.mean().item()
        px1 = (final_error < 1.0).float().mean().item()
        px3 = (final_error < 3.0).float().mean().item()
        px5 = (final_error < 5.0).float().mean().item()
    else:
        epe = px1 = px3 = px5 = 0.0

    metrics = {
        "epe": epe,
        "1px": px1,
        "3px": px3,
        "5px": px5,
        "pivno_init_smooth_l1": init_loss.detach().item(),
        "pivno_recurrent_l1_raw": recurrent_raw_loss.detach().item(),
        "pivno_recurrent_edge_l1": recurrent_edge_loss.detach().item(),
        "disp_grad_loss": gradient_loss.detach().item(),
        "non_degrade_loss": non_degrade_loss.detach().item(),
        "edge_pixel_ratio": edge_ratio.detach().item(),
        "total_loss": total_loss.detach().item(),
    }
    return total_loss, metrics
