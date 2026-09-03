"""PACT2-only recurrent update block with no search-confidence head."""

from __future__ import annotations

import torch

from core.update_pru import (
    MultiPromptUpdateBlock,
    _safe_normalize_disparity,
    interp,
    pool2x,
)


class PACT2FixedRadiusUpdateBlock(MultiPromptUpdateBlock):
    """PRU update that predicts only disparity and convex-upsample weights.

    The shared PACT update block is intentionally left unchanged because its
    three-channel ``delta_info`` output is part of the PACT1 checkpoint and
    MixLap loss contract.
    """

    def __init__(self, args, hidden_dim: int, warp_feat_dim: int,
                 harddim: int) -> None:
        super().__init__(
            args,
            hidden_dim=hidden_dim,
            feat_dim=16,
            volume_dim=1,
            warp_feat_dim=warp_feat_dim,
            harddim=harddim,
            adaptive_search=False,
            use_base_selection=False,
        )

    def forward(self, net, corr, disp, ctx, warp_feat, mono_disp):
        norm_mono, _, _ = _safe_normalize_disparity(mono_disp)
        norm_disp, _, _ = _safe_normalize_disparity(disp)
        structure = self.structure_encoder(ctx, norm_mono - norm_disp)
        motion = self.motion_encoder(corr, disp, warp_feat)

        for i in reversed(range(len(net))):
            if i == len(net) - 1:
                pooled = pool2x(net[i - 1])
                z = self.update[i](torch.cat([net[i], pooled], dim=1))
                pru_out = self.stereo_pru[i](net[i])
                net[i] = (1 - z) * net[i] + z * pru_out
            elif i == 0:
                interp_feat = interp(net[i + 1], net[i])
                z = self.update[i](
                    torch.cat([net[i], structure, motion], dim=1)
                )
                pru_out = self.stereo_pru[i](
                    net[i], interp_feat, structure=structure, motion=motion
                )
                net[i] = (1 - z) * net[i] + z * pru_out
            else:
                pooled = pool2x(net[i - 1])
                interp_feat = interp(net[i + 1], net[i])
                z = self.update[i](torch.cat([net[i], pooled], dim=1))
                pru_out = self.stereo_pru[i](net[i], interp_feat)
                net[i] = (1 - z) * net[i] + z * pru_out

        delta_disp = self.disp_head(net[0])
        mask = 0.25 * self.mask(net[0])
        return net, delta_disp, mask


__all__ = ["PACT2FixedRadiusUpdateBlock"]
