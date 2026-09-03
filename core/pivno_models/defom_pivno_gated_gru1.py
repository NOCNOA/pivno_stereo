"""Gated encoded PIVNO refinement with ordinary 1x1 outer ConvGRUs.

This is an isolated ablation of ``defom_pivno_gated_gru3``.  It preserves the
PIVNO initializer, sampled-right encoding, scale gate, fusion path, recurrent
topology, and prediction heads, and changes only the spatial kernel used by
the three outer ConvGRUs from 3x3 to 1x1.
"""

from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as GatedGRU3DEFOMStereo,
)
from core.update import BasicMultiUpdateBlock


class DEFOMStereo(GatedGRU3DEFOMStereo):
    """Checkpoint-isolated 1x1-ConvGRU counterpart to the GRU3 model."""

    MODEL_VARIANT = "defom_pivno_gated_gru1"
    GRU_KERNEL_SIZE = 1

    def __init__(self, args):
        super().__init__(args)

        # The parent constructs the complete gated/encoded GRU3 model. Replace
        # only its final update block so every non-GRU feature contract remains
        # identical while the old model class and checkpoint path stay intact.
        self.update_block = BasicMultiUpdateBlock(
            self.args,
            hidden_dims=args.hidden_dims,
        )
