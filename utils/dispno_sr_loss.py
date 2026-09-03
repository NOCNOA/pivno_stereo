"""Loss for the isolated DispNO+SR stereo model."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_disparity(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.unsqueeze(1)
    if value.ndim == 4 and value.shape[1] == 1:
        return value
    raise ValueError(
        f"expected scalar disparity [B,H,W] or [B,1,H,W], got {tuple(value.shape)}"
    )


def dispno_sr_sequence_loss(
    predictions: Sequence[torch.Tensor],
    disparity_gt: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_disp: float = 768.0,
    gamma: float = 0.9,
    smooth_l1_beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervise all internal DispNO predictions at full resolution."""
    if not predictions:
        raise ValueError("predictions must contain at least one disparity")
    disparity_gt = _as_disparity(disparity_gt).float()
    valid = _as_disparity(valid)
    finite_gt = torch.isfinite(disparity_gt)
    safe_gt = torch.where(finite_gt, disparity_gt, torch.zeros_like(disparity_gt))
    valid_mask = (
        (valid >= 0.5)
        & finite_gt
        & (safe_gt >= 0.0)
        & (safe_gt < float(max_disp))
    )
    valid_float = valid_mask.float()
    denominator = valid_float.sum().clamp_min(1.0)

    adjusted_gamma = float(gamma) ** (15.0 / len(predictions))
    loss = safe_gt.new_zeros(())
    checked = []
    for index, prediction in enumerate(predictions):
        prediction = _as_disparity(prediction)
        if prediction.shape != safe_gt.shape:
            raise ValueError(
                f"prediction[{index}] must have shape {tuple(safe_gt.shape)}, "
                f"got {tuple(prediction.shape)}"
            )
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(
                f"prediction[{index}] contains non-finite values"
            )
        weight = adjusted_gamma ** (len(predictions) - index - 1)
        per_pixel = F.smooth_l1_loss(
            prediction.float(),
            safe_gt,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
        loss = loss + weight * (
            per_pixel * valid_float
        ).sum() / denominator
        checked.append(prediction.float())

    final_error = (checked[-1] - safe_gt).abs()[valid_mask]
    if final_error.numel() == 0:
        epe = out1 = out3 = out5 = 0.0
    else:
        epe = final_error.mean().item()
        out1 = (final_error > 1.0).float().mean().item()
        out3 = (final_error > 3.0).float().mean().item()
        out5 = (final_error > 5.0).float().mean().item()
    return loss, {
        "epe": epe,
        "out1": out1,
        "out3": out3,
        "out5": out5,
        "valid_pixels": int(valid_mask.sum().item()),
        "sequence_smooth_l1": float(loss.detach()),
    }

