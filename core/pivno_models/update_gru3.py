"""PIVNO-only recurrent update block with IGEV++-style 3x3 ConvGRUs."""

from core.update import BasicMultiUpdateBlock, ConvGRU


class IGEVStyleBasicMultiUpdateBlock(BasicMultiUpdateBlock):
    """Keep the existing PIVNO update topology but use 3x3 GRU gates."""

    GRU_KERNEL_SIZE = 3

    def __init__(self, args, hidden_dims=(128, 128, 128)):
        super().__init__(args, hidden_dims=list(hidden_dims))
        encoder_output_dim = 128
        kernel_size = self.GRU_KERNEL_SIZE

        self.gru08 = ConvGRU(
            hidden_dims[2],
            encoder_output_dim
            + hidden_dims[1] * (args.n_gru_layers > 1),
            kernel_size,
        )
        self.gru16 = ConvGRU(
            hidden_dims[1],
            hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2],
            kernel_size,
        )
        self.gru32 = ConvGRU(
            hidden_dims[0],
            hidden_dims[1],
            kernel_size,
        )

