"""Strict C32/GWC4 ablation with only the outer ConvGRU kernel changed.

The completed gated-GRU3 checkpoint was trained with 32 low-level feature
channels, four GWC groups, and 16 encoded match channels.  This isolated model
freezes that complete feature/fusion contract and exposes only a 1x1 versus
3x3 choice for the three outer multi-scale ConvGRUs.

Existing GRU1/GRU3 classes are intentionally left untouched so their
checkpoints keep their original meaning.
"""

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    CheckpointCompatibleGWC4GatedGRU3,
)
from core.update import BasicMultiUpdateBlock


class DEFOMStereo(CheckpointCompatibleGWC4GatedGRU3):
    """C32/GWC4 gated PIVNO whose sole ablation variable is GRU kernel size."""

    MODEL_VARIANT = "defom_pivno_gated_gru_kernel_ablation"
    ALLOWED_GRU_KERNEL_SIZES = (1, 3)
    LOW_FEATURE_DIM = 32
    MATCH_NUM_GROUPS = 4
    MATCH_ENCODED_CHANNELS = 16

    def __init__(self, args):
        kernel_size = getattr(args, "pivno_gru_kernel_size", None)
        if kernel_size is None:
            raise ValueError(
                "pivno_gru_kernel_size is required for the strict GRU "
                "ablation; choose 1 or 3"
            )
        kernel_size = int(kernel_size)
        if kernel_size not in self.ALLOWED_GRU_KERNEL_SIZES:
            raise ValueError(
                "pivno_gru_kernel_size must be 1 or 3, got "
                f"{kernel_size}"
            )

        super().__init__(args)
        self.GRU_KERNEL_SIZE = kernel_size

        # The checkpoint-compatible parent already constructs the exact 3x3
        # block used by the completed baseline.  For the 1x1 arm, replace only
        # that block with the ordinary RAFT-Stereo update block.  Encoder,
        # motion encoder, prediction head, mask head, topology, and all feature
        # paths remain identical; only the nine z/r/q gate weight kernels have
        # different spatial shapes.
        if kernel_size == 1:
            baseline_update_block = self.update_block
            ablation_update_block = BasicMultiUpdateBlock(
                self.args,
                hidden_dims=args.hidden_dims,
            )
            baseline_state = baseline_update_block.state_dict()
            ablation_state = ablation_update_block.state_dict()
            for name, value in baseline_state.items():
                if ablation_state[name].shape == value.shape:
                    ablation_state[name].copy_(value)
            self.update_block = ablation_update_block
