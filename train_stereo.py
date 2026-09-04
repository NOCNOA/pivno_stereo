from __future__ import print_function, division
import os
import sys
import logging
import argparse
import numpy as np
import time
from itertools import islice
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from utils.utils import *
from utils.pact_pivno_loss import pact_pivno_sequence_loss
from core.pivno_models.defom_pact_pivno import DEFOMStereo as PACTPIVNODEFOMStereo
from core.pivno_models.defom_pivno import DEFOMStereo as PIVNODEFOMStereo
from core.pivno_models.defom_pivno_mobilenetv2 import DEFOMStereo as MobileNetV2PIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated import DEFOMStereo as GatedPIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated_gru1 import DEFOMStereo as GatedGRU1PIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated_gru3 import DEFOMStereo as GatedGRU3PIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated_gru3_gwc_only import DEFOMStereo as GatedGRU3GWCOnlyPIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated_gru_kernel_ablation import DEFOMStereo as GatedGRUKernelAblationPIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import DEFOMStereo as GatedGRU3GWC4MaskSRPIVNODEFOMStereo
from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_rgb_sr import DEFOMStereo as GatedGRU3GWC4MaskRGBSRPIVNODEFOMStereo
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3 import DEFOMStereo as GWC4Enc16ConcatGRU3PIVNODEFOMStereo
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3_mask_sr import DEFOMStereo as GWC4Enc16ConcatGRU3MaskSRPIVNODEFOMStereo
try:
    from core.defom_low_reso_volume import DEFOMStereo as LegacyDEFOMStereo
except ModuleNotFoundError:
    LegacyDEFOMStereo = None
#from core.defom_GRU_2multi import DEFOMStereo
#from core.defom_onlygwc import DEFOMStereo
#from core.defom_pru import DEFOMStereo
from evaluate_stereo import validate_things, count_parameters
import core.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def _infer_pivno_input_channels(checkpoint):
    """Infer legacy PIVNO RGB/grayscale input width from metadata or weights."""
    config = checkpoint.get('model_config', {}) if isinstance(checkpoint, dict) else {}
    if isinstance(config, dict) and 'pivno_input_channels' in config:
        channels = int(config['pivno_input_channels'])
    else:
        state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        key = next(
            (
                name for name in state
                if name.endswith('pivno.snet.conv1.weight')
            ),
            None,
        )
        if key is None:
            raise ValueError(
                'cannot infer PIVNO input channels: missing '
                'pivno.snet.conv1.weight'
            )
        channels = int(state[key].shape[1])
    if channels not in (1, 3):
        raise ValueError(
            f'PIVNO checkpoint input channels must be 1 or 3, got {channels}'
        )
    return channels


class OffsetBatchSampler:
    """Skip completed batch indices without loading or augmenting their data."""

    def __init__(self, batch_sampler, offset):
        self.batch_sampler = batch_sampler
        self.offset = int(offset)
        if not 0 <= self.offset <= len(self.batch_sampler):
            raise ValueError(
                f"batch sampler offset {self.offset} is outside "
                f"[0, {len(self.batch_sampler)}]"
            )

    def __iter__(self):
        return islice(iter(self.batch_sampler), self.offset, None)

    def __len__(self):
        return len(self.batch_sampler) - self.offset


def train(args):

    model_name = getattr(args, 'model', 'legacy')
    use_pact_smd = model_name == 'pact_smd'
    use_pact_smd_post = model_name == 'pact_smd_post'
    use_pact_bilap = model_name == 'pact_bilap_gru'
    use_pact_pivno = model_name == 'pact_pivno'
    use_mobilenetv2_pivno = model_name == 'defom_pivno_mobilenetv2'
    use_concat_gru3_mask_sr = (
        model_name == 'defom_pivno_gwc4_enc16_concat_gru3_mask_sr'
    )
    use_gated_gru3_mask_sr = (
        model_name == 'defom_pivno_gated_gru3_gwc4_mask_sr'
    )
    use_gated_gru3_mask_rgb_sr = (
        model_name == 'defom_pivno_gated_gru3_gwc4_mask_rgb_sr'
    )
    use_gated_gru3_mask_rgb_hidden_sr = (
        model_name == 'defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr'
    )
    use_gated_gru3_mask_last_delta_sr = (
        model_name == 'defom_pivno_gated_gru3_gwc4_mask_last_delta_sr'
    )
    use_gated_gru3_last_delta_direct_sr = (
        model_name == 'defom_pivno_gated_gru3_gwc4_last_delta_direct_sr'
    )
    use_gated_gru3_sr = (
        use_gated_gru3_mask_sr
        or use_gated_gru3_mask_rgb_sr
        or use_gated_gru3_mask_rgb_hidden_sr
        or use_gated_gru3_mask_last_delta_sr
        or use_gated_gru3_last_delta_direct_sr
    )
    use_mask_sr = use_concat_gru3_mask_sr or use_gated_gru3_sr
    use_gated_gru3_pivno = model_name == 'defom_pivno_gated_gru3'
    use_gated_gru3_gwc_only_pivno = (
        model_name == 'defom_pivno_gated_gru3_gwc_only'
    )
    use_gated_gru1_pivno = model_name == 'defom_pivno_gated_gru1'
    use_gated_gru_kernel_ablation = (
        model_name == 'defom_pivno_gated_gru_kernel_ablation'
    )
    use_gated_gru3_family = (
        use_gated_gru3_pivno
        or use_gated_gru3_gwc_only_pivno
        or use_gated_gru3_sr
    )
    use_concat_gru3_pivno = (
        model_name == 'defom_pivno_gwc4_enc16_concat_gru3'
    )
    use_concat_gru3_family = (
        use_concat_gru3_pivno or use_concat_gru3_mask_sr
    )
    use_encoded_gru_pivno = (
        use_gated_gru1_pivno
        or use_gated_gru_kernel_ablation
        or use_gated_gru3_family
        or use_concat_gru3_family
    )
    use_gated_pivno = model_name in (
        'defom_pivno_gated',
        'defom_pivno_gated_gru1',
        'defom_pivno_gated_gru3',
        'defom_pivno_gated_gru3_gwc_only',
        'defom_pivno_gated_gru_kernel_ablation',
    )
    use_scale_gate_pivno = use_gated_pivno or use_gated_gru3_sr
    use_defom_pivno = model_name in (
        'defom_pivno',
        'defom_pivno_mobilenetv2',
        'defom_pivno_gated',
        'defom_pivno_gated_gru1',
        'defom_pivno_gated_gru3',
        'defom_pivno_gated_gru3_gwc_only',
        'defom_pivno_gated_gru_kernel_ablation',
        'defom_pivno_gwc4_enc16_concat_gru3',
        'defom_pivno_gwc4_enc16_concat_gru3_mask_sr',
        'defom_pivno_gated_gru3_gwc4_mask_sr',
        'defom_pivno_gated_gru3_gwc4_mask_rgb_sr',
        'defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr',
        'defom_pivno_gated_gru3_gwc4_mask_last_delta_sr',
        'defom_pivno_gated_gru3_gwc4_last_delta_direct_sr',
    )
    use_pivno = use_pact_pivno or use_defom_pivno
    use_pact = model_name in ('pact', 'pact_smd', 'pact_smd_post', 'pact_bilap_gru', 'pact2', 'pact2_gev')
    use_pact2_gev = model_name == 'pact2_gev'
    use_pact2 = model_name in ('pact2', 'pact2_gev')
    configured_eval_max_disp = getattr(args, 'eval_max_disp', None)
    if configured_eval_max_disp is None:
        validation_max_disp = float(args.max_disp)
    elif configured_eval_max_disp <= 0:
        validation_max_disp = None
    else:
        validation_max_disp = float(configured_eval_max_disp)
    benchmark_mode = args.benchmark_steps > 0
    if benchmark_mode:
        args.num_steps = args.benchmark_steps

    if args.launcher == 'none':
        args.distributed = False
        rank, world_size = 0, 1
        is_main_process = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True
        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        rank, world_size = get_dist_info()
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        is_main_process = rank == 0

        global_batch_size = args.batch_size
        if global_batch_size % world_size != 0:
            raise ValueError(
                f'Global batch size {global_batch_size} must be divisible by '
                f'DDP world size {world_size}'
            )
        args.batch_size = global_batch_size // world_size
        args.gpu_ids = range(world_size)
        setup_for_distributed(is_main_process)

    global_batch_size = args.batch_size * world_size if args.distributed else args.batch_size
    benchmark_peak_memory = None
    if benchmark_mode and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if is_main_process:
        mask_description = (
            "disabled" if validation_max_disp is None
            else f"0 <= disp_gt < {validation_max_disp:g}"
        )
        logging.info(
            "Validation disparity mask: %s (model max_disp=%s)",
            mask_description,
            args.max_disp,
        )

    seed_everything(args.seed + rank)

    if use_gated_gru3_mask_rgb_sr:
        model_cls = GatedGRU3GWC4MaskRGBSRPIVNODEFOMStereo
    elif use_gated_gru3_mask_sr:
        model_cls = GatedGRU3GWC4MaskSRPIVNODEFOMStereo
    elif use_concat_gru3_mask_sr:
        model_cls = GWC4Enc16ConcatGRU3MaskSRPIVNODEFOMStereo
    elif use_concat_gru3_pivno:
        model_cls = GWC4Enc16ConcatGRU3PIVNODEFOMStereo
    elif use_gated_gru1_pivno:
        model_cls = GatedGRU1PIVNODEFOMStereo
    elif use_gated_gru_kernel_ablation:
        model_cls = GatedGRUKernelAblationPIVNODEFOMStereo
    elif use_gated_gru3_pivno:
        model_cls = GatedGRU3PIVNODEFOMStereo
    elif use_gated_gru3_gwc_only_pivno:
        model_cls = GatedGRU3GWCOnlyPIVNODEFOMStereo
    elif use_gated_pivno:
        model_cls = GatedPIVNODEFOMStereo
    elif use_mobilenetv2_pivno:
        model_cls = MobileNetV2PIVNODEFOMStereo
    elif use_defom_pivno:
        model_cls = PIVNODEFOMStereo
    elif use_pact_pivno:
        model_cls = PACTPIVNODEFOMStereo
    else:
        if LegacyDEFOMStereo is None:
            raise RuntimeError(
                "legacy training requires core/defom_low_reso_volume.py, "
                "which is not present in this portable checkout"
            )
        model_cls = LegacyDEFOMStereo
    model = model_cls(args).to(device)
    model_config = {
        'model': args.model,
        'state_mode': 'dual_laplace_distribution' if use_pact_bilap else 'single_current_disp' if use_pact else 'pivno_single_current_disp' if use_pivno else 'legacy',
        'model_variant': (
            model_cls.MODEL_VARIANT if (use_pact2 or use_pact_smd or use_pact_smd_post or use_pact_bilap)
            else 'pact1' if use_pact else model_name if use_pivno else 'legacy'
        ),
        'feature_backbone': (
            model_cls.FEATURE_BACKBONE
            if (use_pact2 or use_mobilenetv2_pivno) else None
        ),
        'dinov2_encoder': args.dinov2_encoder,
        'idepth_scale': float(args.idepth_scale),
        'max_disp': int(args.max_disp),
        'n_downsample': int(args.n_downsample),
        'n_gru_layers': int(args.n_gru_layers),
        'hidden_dims': list(args.hidden_dims),
        'pact_checkpoint_corr': bool(getattr(args, 'pact_checkpoint_corr', True)),
        'pact_fp32_stereo': bool(getattr(args, 'pact_fp32_stereo', True)),
        'pact_fp32_update': bool(getattr(args, 'pact_fp32_update', True)),
        'pact_mid_refine_iters': int(getattr(args, 'pact_mid_refine_iters', 1)),
    }
    if use_pact2:
        model_config.update({
            'pact_sampling_layout': model_cls.SAMPLING_LAYOUT,
            'pact_fixed_radius': model_cls.FIXED_RADIUS_QUARTER,
        })
        if use_pact2_gev:
            model_config['pact_gev_mode'] = str(args.pact_gev_mode)
    elif use_pact:
        model_config.update({
            'pact_sampling_layout': str(
                getattr(args, 'pact_sampling_layout', 'legacy9')
            ),
            'pact_min_radius': float(getattr(args, 'pact_min_radius', 1.0)),
            'pact_max_radius': float(getattr(args, 'pact_max_radius', 8.0)),
            'pact_mid_delta_scale': float(
                getattr(args, 'pact_mid_delta_scale', 1.0)
            ),
        })
        if use_pact_bilap:
            model_config.update({
                'pact_smd_mode_threshold': float(args.pact_smd_mode_threshold),
                'bilap_ablation': str(args.bilap_ablation),
                'bilap_width_levels': 2,
                'bilap_match_channels': 32,
                'bilap_gwc_groups': 4,
                'bilap_checkpoint_update': bool(args.bilap_checkpoint_update),
                'bilap_init': str(args.bilap_init),
                'bilap_init_delta': float(args.bilap_init_delta),
                'bilap_init_scale': float(args.bilap_init_scale),
                'bilap_separate_mode_gru': bool(args.bilap_separate_mode_gru),
                'bilap_lookup_mode': str(args.bilap_lookup_mode),
                'bilap_q_min': float(args.bilap_q_min),
                'bilap_q_max': float(args.bilap_q_max),
                'bilap_q_scale': float(args.bilap_q_scale),
                'bilap_diversity_weight': float(args.bilap_diversity_weight),
                'bilap_nll_region': str(args.bilap_nll_region),
            })
        elif use_pact_smd_post:
            model_config.update({
                'pact_smd_stage': 'post',
                'pact_smd_mode_threshold': float(args.pact_smd_mode_threshold),
                'pact_smd_post_mode_threshold': float(
                    args.pact_smd_post_mode_threshold
                ),
                'pact_smd_post_local_radius': float(
                    args.pact_smd_post_local_radius
                ),
                'pact_smd_post_broad_radius_min': float(
                    args.pact_smd_post_broad_radius_min
                ),
                'pact_smd_post_broad_radius_max': float(
                    args.pact_smd_post_broad_radius_max
                ),
            })
        elif use_pact_smd:
            model_config.update({
                'pact_smd_stage': str(args.pact_smd_stage),
                'pact_smd_grad_iters': int(args.pact_smd_grad_iters),
                'pact_smd_mode_threshold': float(args.pact_smd_mode_threshold),
                'pact_smd_head_lr': float(args.pact_smd_head_lr),
                'pact_smd_adapt_lr': float(args.pact_smd_adapt_lr),
                'pact_smd_head_sequence_weight': float(
                    args.pact_smd_head_sequence_weight
                ),
                'pact_smd_nll_head_weight': float(args.pact_smd_nll_head_weight),
                'pact_smd_nll_joint_weight': float(args.pact_smd_nll_joint_weight),
                'pact_smd_nll_full_weight': float(args.pact_smd_nll_full_weight),
                'pact_smd_base_aux_weight': float(args.pact_smd_base_aux_weight),
                'pact_smd_selection_weight': float(args.pact_smd_selection_weight),
                'pact_smd_guard_weight': float(args.pact_smd_guard_weight),
            })
    elif use_pivno:
        model_config.update({
            'pivno_input_channels': int(model.pivno.input_channels),
            'pivno_input_range': 'rgb_0_1',
        })
        if use_mobilenetv2_pivno:
            model_config.update({
                'pivno_feature_encoder': model_cls.FEATURE_BACKBONE,
                'pivno_imagenet_pretrained': False,
            })
        if use_mask_sr:
            model_config.update({
                'model_variant': model.MODEL_VARIANT,
                'pivno_mask_sr_base_model': model.BASE_MODEL_VARIANT,
                'pivno_mask_sr_stage': model.sr_stage,
                'pivno_mask_sr_final_only': True,
                'pivno_mask_sr_feature_channels': int(
                    model.sr_head.FEATURE_CHANNELS
                ),
                'pivno_mask_sr_input_channels': int(
                    model.sr_head.INPUT_CHANNELS
                ),
                'pivno_mask_sr_residual_max': float(
                    model.sr_head.residual_max
                ),
                'pivno_mask_sr_output': (
                    model.sr_head.OUTPUT_MODE
                    if (
                        use_gated_gru3_mask_last_delta_sr
                        or use_gated_gru3_last_delta_direct_sr
                    )
                    else 'bounded_full_resolution_delta_d'
                ),
            })
            if (
                use_gated_gru3_mask_rgb_sr
                or use_gated_gru3_mask_rgb_hidden_sr
                or use_gated_gru3_mask_last_delta_sr
                or use_gated_gru3_last_delta_direct_sr
            ):
                model_config['pivno_mask_sr_feature_source'] = (
                    model.sr_head.FEATURE_SOURCE
                )
            if use_gated_gru3_mask_last_delta_sr:
                model_config.update({
                    'pivno_mask_sr_weight_mode': model.sr_head.WEIGHT_MODE,
                    'pivno_mask_sr_max_delta_disp_low': float(
                        model.sr_head.max_delta_disp_low
                    ),
                })
            if use_gated_gru3_last_delta_direct_sr:
                model_config.update({
                    'pivno_mask_sr_upsample_mode': model.sr_head.UPSAMPLE_MODE,
                    'pivno_mask_sr_max_delta_disp_low': float(
                        model.sr_head.max_delta_disp_low
                    ),
                    'pivno_mask_sr_max_delta_disp_hr': float(
                        model.sr_head.max_delta_disp_hr
                    ),
                    'pivno_mask_sr_final_composition': (
                        'previous_iteration_upsampled_disp_plus_direct_delta'
                    ),
                })
            if use_gated_gru3_mask_rgb_hidden_sr:
                pretrained_lr = getattr(
                    args, 'pivno_mask_sr_pretrained_lr', None
                )
                model_config['pivno_mask_sr_pretrained_lr'] = float(
                    0.1 * args.lr if pretrained_lr is None
                    else pretrained_lr
                )
        if use_scale_gate_pivno:
            model_config.update({
                'pivno_scale_gate': model.SCALE_GATE_MODE,
                'pivno_scale_gate_identity_init': True,
                'corr_radius': int(args.corr_radius),
                'pivno_base_lr': float(args.lr),
                'pivno_gate_lr': float(
                    args.lr if args.pivno_gate_lr is None
                    else args.pivno_gate_lr
                ),
            })
        if use_encoded_gru_pivno:
            model_config.update({
                'pivno_gru_kernel_size': int(model.GRU_KERNEL_SIZE),
                'pivno_right_sample_encoding': (
                    model.RIGHT_SAMPLE_ENCODING
                ),
                'pivno_match_num_groups': int(model.MATCH_NUM_GROUPS),
                'pivno_match_encoded_channels': int(
                    model.MATCH_ENCODED_CHANNELS
                ),
                'pivno_amp_policy': model.AMP_POLICY,
            })
        if use_gated_gru_kernel_ablation:
            model_config['pivno_low_feature_dim'] = int(
                model.LOW_FEATURE_DIM
            )
        if use_concat_gru3_family:
            model_config.update({
                'pivno_fusion_mode': model.FUSION_MODE,
                'pivno_low_feature_dim': int(model.LOW_FEATURE_DIM),
                'pivno_scale_gate': 'none',
            })
        if use_gated_gru3_sr:
            model_config['pivno_low_feature_dim'] = int(
                model.LOW_FEATURE_DIM
            )
    print("Parameter Count: %d, Trainable: %d" % count_parameters(model))
    if use_pact_smd_post:
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        invalid_trainable = [
            name for name in trainable_names
            if not name.startswith('final_smd_head.')
        ]
        if not trainable_names or invalid_trainable:
            raise ValueError(
                'PACT-SMD-post must train only final_smd_head; '
                f'trainable={trainable_names}'
            )
        logging.info(
            'PACT-SMD-post frozen training contract: %d trainable tensors, '
            'all under final_smd_head',
            len(trainable_names),
        )
    if use_mask_sr and model.sr_stage == 'head':
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        invalid_trainable = [
            name for name in trainable_names
            if not name.startswith('sr_head.')
        ]
        if not trainable_names or invalid_trainable:
            raise ValueError(
                'PIVNO-mask-SR head stage must train only sr_head; '
                f'trainable={trainable_names}'
            )
        logging.info(
            'PIVNO-mask-SR frozen training contract: %d trainable '
            'tensors, all under sr_head',
            len(trainable_names),
        )

    if args.distributed:
        if use_pact:
            # PACT uses a small per-rank batch. Frozen BN avoids SyncBN
            # collectives in every recurrent step.
            model.freeze_bn()
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=not use_pact,
            # Bucket-view aliasing benchmarks slower on the PCIe-only 3090
            # topology used here, despite saving one gradient copy.
            gradient_as_bucket_view=not use_pact,
            static_graph=use_pact,
        )
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model_without_ddp = model.module
        else:
            model_without_ddp = model

        model_without_ddp.freeze_bn()  # legacy behavior; refreshed below for PACT

    start_epoch = 0
    start_step = 0
    resume_batch_in_epoch = None
    resume_scaler_state = None
    optimizer, scheduler = fetch_optimizer(args, model)

    if args.resume_ckpt:
        assert args.resume_ckpt.endswith(".pth")
        logging.info("Loading checkpoint: %s" % args.resume_ckpt)
        # Load once on CPU and let load_state_dict move tensors to their
        # parameter devices.  Keeping the full model+Adam checkpoint on every
        # GPU wastes roughly the checkpoint size and is especially costly now
        # that the stable stereo feature path uses FP32 activations.
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        post_source_model = None
        if use_pact_bilap:
            checkpoint_config = checkpoint.get('model_config') if isinstance(checkpoint, dict) else None
            expected_bilap = {
                'model': 'pact_bilap_gru',
                'model_variant': model_cls.MODEL_VARIANT,
                'state_mode': 'dual_laplace_distribution',
                'max_disp': int(args.max_disp),
                'bilap_ablation': str(args.bilap_ablation),
                'bilap_width_levels': 2,
                'bilap_match_channels': 32,
                'bilap_gwc_groups': 4,
                'bilap_checkpoint_update': bool(args.bilap_checkpoint_update),
                'bilap_init': str(args.bilap_init),
                'bilap_separate_mode_gru': bool(args.bilap_separate_mode_gru),
                'bilap_lookup_mode': str(args.bilap_lookup_mode),
                'bilap_nll_region': str(args.bilap_nll_region),
            }
            mismatches = {key: (None if checkpoint_config is None else checkpoint_config.get(key), value) for key, value in expected_bilap.items() if checkpoint_config is None or checkpoint_config.get(key) != value}
            if mismatches:
                raise ValueError(f'PACT-BiLap-GRU resume checkpoint/configuration mismatch: {mismatches}. Start from scratch or use a matching BiLap checkpoint.')
        if use_pact_smd_post:
            checkpoint_config = (
                checkpoint.get('model_config')
                if isinstance(checkpoint, dict) else None
            )
            if not isinstance(checkpoint_config, dict):
                raise ValueError(
                    'PACT-SMD-post requires model_config metadata from a '
                    'trained Full or post checkpoint'
                )
            post_source_model = checkpoint_config.get('model')
            if post_source_model not in ('pact_smd', 'pact_smd_post'):
                raise ValueError(
                    'PACT-SMD-post must reuse pact_smd Full weights or resume '
                    f'a post checkpoint, got model={post_source_model!r}'
                )
            expected_shared = {
                'max_disp': int(args.max_disp),
                'dinov2_encoder': args.dinov2_encoder,
                'n_downsample': int(args.n_downsample),
                'n_gru_layers': int(args.n_gru_layers),
                'hidden_dims': list(args.hidden_dims),
                'pact_sampling_layout': str(args.pact_sampling_layout),
                'pact_min_radius': float(args.pact_min_radius),
                'pact_max_radius': float(args.pact_max_radius),
                'pact_mid_delta_scale': float(args.pact_mid_delta_scale),
                'pact_smd_mode_threshold': float(args.pact_smd_mode_threshold),
            }
            mismatches = {
                key: (checkpoint_config.get(key), expected)
                for key, expected in expected_shared.items()
                if checkpoint_config.get(key) != expected
            }
            if post_source_model == 'pact_smd':
                expected_source = {
                    'model_variant': PACTSMDDEFOMStereo.MODEL_VARIANT,
                    'pact_smd_stage': 'full',
                }
            else:
                expected_source = {
                    'model_variant': model_cls.MODEL_VARIANT,
                    'pact_smd_stage': 'post',
                    'pact_smd_post_mode_threshold': float(
                        args.pact_smd_post_mode_threshold
                    ),
                    'pact_smd_post_local_radius': float(
                        args.pact_smd_post_local_radius
                    ),
                    'pact_smd_post_broad_radius_min': float(
                        args.pact_smd_post_broad_radius_min
                    ),
                    'pact_smd_post_broad_radius_max': float(
                        args.pact_smd_post_broad_radius_max
                    ),
                }
            mismatches.update({
                key: (checkpoint_config.get(key), expected)
                for key, expected in expected_source.items()
                if checkpoint_config.get(key) != expected
            })
            if mismatches:
                raise ValueError(
                    'PACT-SMD-post checkpoint/configuration mismatch: '
                    f'{mismatches}'
                )
        if use_pact_smd and isinstance(checkpoint, dict):
            checkpoint_config = checkpoint.get('model_config')
            if isinstance(checkpoint_config, dict):
                checkpoint_model = checkpoint_config.get('model', 'pact')
                if checkpoint_model not in ('pact', 'pact_smd'):
                    raise ValueError(
                        'PACT-SMD can initialize only from pact or pact_smd, '
                        f'got model={checkpoint_model!r}'
                    )
                expected_shared = {
                    'max_disp': int(args.max_disp),
                    'dinov2_encoder': args.dinov2_encoder,
                    'n_downsample': int(args.n_downsample),
                    'n_gru_layers': int(args.n_gru_layers),
                    'hidden_dims': list(args.hidden_dims),
                    'pact_sampling_layout': str(args.pact_sampling_layout),
                    'pact_min_radius': float(args.pact_min_radius),
                    'pact_max_radius': float(args.pact_max_radius),
                    'pact_mid_delta_scale': float(args.pact_mid_delta_scale),
                }
                legacy_defaults = {
                    'pact_sampling_layout': 'legacy9',
                    'pact_min_radius': 1.0,
                    'pact_max_radius': 8.0,
                    'pact_mid_delta_scale': 1.0,
                }
                mismatches = {
                    key: (
                        checkpoint_config.get(key, legacy_defaults.get(key)),
                        expected,
                    )
                    for key, expected in expected_shared.items()
                    if checkpoint_config.get(key, legacy_defaults.get(key))
                    != expected
                }
                if checkpoint_model == 'pact_smd':
                    expected_smd = {
                        'model_variant': model_cls.MODEL_VARIANT,
                        'pact_smd_mode_threshold': float(
                            args.pact_smd_mode_threshold
                        ),
                    }
                    mismatches.update({
                        key: (checkpoint_config.get(key), expected)
                        for key, expected in expected_smd.items()
                        if checkpoint_config.get(key) != expected
                    })
                if mismatches:
                    raise ValueError(
                        'PACT-SMD checkpoint/configuration mismatch: '
                        f'{mismatches}'
                    )
        if use_pact2:
            checkpoint_config = (
                checkpoint.get('model_config')
                if isinstance(checkpoint, dict) else None
            )
            expected_pact2_config = {
                'model': args.model,
                'model_variant': model_cls.MODEL_VARIANT,
                'pact_sampling_layout': model_cls.SAMPLING_LAYOUT,
                'pact_fixed_radius': model_cls.FIXED_RADIUS_QUARTER,
            }
            if use_pact2_gev:
                expected_pact2_config['pact_gev_mode'] = str(
                    args.pact_gev_mode
                )
            mismatches = {
                key: (None if checkpoint_config is None
                      else checkpoint_config.get(key), expected)
                for key, expected in expected_pact2_config.items()
                if checkpoint_config is None
                or checkpoint_config.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    "PACT2 resume checkpoint predates the fixed-radius "
                    f"architecture or has incompatible metadata: {mismatches}. "
                    "Start a new training run instead of resuming it."
                )
        checkpoint_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
        checkpoint_model_for_sr = (
            checkpoint.get('model_config', {}).get('model')
            if isinstance(checkpoint, dict) else None
        )
        if (
            use_gated_gru1_pivno
            and checkpoint_model_for_sr != model_name
        ):
            raise ValueError(
                f'{model_name} can resume only its own 1x1-GRU checkpoint, '
                f'got {checkpoint_model_for_sr!r}. A GRU3 checkpoint has '
                'incompatible 3x3 gate weights.'
            )
        if use_gated_gru_kernel_ablation:
            checkpoint_config = (
                checkpoint.get('model_config', {})
                if isinstance(checkpoint, dict) else {}
            )
            checkpoint_model = checkpoint_config.get('model')
            checkpoint_kernel = checkpoint_config.get(
                'pivno_gru_kernel_size'
            )
            expected_kernel = int(model_without_ddp.GRU_KERNEL_SIZE)
            if (
                checkpoint_model != model_name
                or checkpoint_kernel != expected_kernel
            ):
                raise ValueError(
                    f'{model_name} kernel={expected_kernel} can resume only '
                    'its own matching-kernel checkpoint, got '
                    f'model={checkpoint_model!r}, '
                    f'kernel={checkpoint_kernel!r}'
                )
        sr_from_base_concat_gru3 = (
            use_concat_gru3_mask_sr
            and checkpoint_model_for_sr
            == 'defom_pivno_gwc4_enc16_concat_gru3'
        )
        sr_from_base_gated_gru3 = (
            use_gated_gru3_sr
            and checkpoint_model_for_sr == 'defom_pivno_gated_gru3'
        )
        fusion_from_hidden_sr = (
            use_gated_gru3_mask_rgb_hidden_sr
            and checkpoint_model_for_sr
            == 'defom_pivno_gated_gru3_gwc4_mask_sr'
        )
        concat_from_gated_gru3 = (
            use_concat_gru3_pivno
            and isinstance(checkpoint, dict)
            and checkpoint.get('model_config', {}).get('model')
            == 'defom_pivno_gated_gru3'
        )
        load_strict = args.strict_resume and not (
            use_pact_smd_post and post_source_model == 'pact_smd'
        ) and not concat_from_gated_gru3 \
            and not sr_from_base_concat_gru3 \
            and not sr_from_base_gated_gru3 \
            and not fusion_from_hidden_sr
        incompatible = model_without_ddp.load_state_dict(
            checkpoint_state, strict=load_strict
        )
        if use_concat_gru3_mask_sr:
            checkpoint_config = (
                checkpoint.get('model_config', {})
                if isinstance(checkpoint, dict) else {}
            )
            checkpoint_model = checkpoint_config.get('model')
            base_model_name = 'defom_pivno_gwc4_enc16_concat_gru3'
            if checkpoint_model not in (base_model_name, model_name):
                raise ValueError(
                    f'{model_name} can initialize only from {base_model_name} '
                    f'or resume itself, got {checkpoint_model!r}'
                )
            expected_shared = {
                'max_disp': int(args.max_disp),
                'n_downsample': int(args.n_downsample),
                'n_gru_layers': int(args.n_gru_layers),
                'hidden_dims': list(args.hidden_dims),
                'pivno_input_channels': 3,
                'pivno_gru_kernel_size': int(
                    model_without_ddp.GRU_KERNEL_SIZE
                ),
                'pivno_right_sample_encoding': (
                    model_without_ddp.RIGHT_SAMPLE_ENCODING
                ),
                'pivno_match_num_groups': int(
                    model_without_ddp.MATCH_NUM_GROUPS
                ),
                'pivno_match_encoded_channels': int(
                    model_without_ddp.MATCH_ENCODED_CHANNELS
                ),
                'pivno_fusion_mode': model_without_ddp.FUSION_MODE,
                'pivno_low_feature_dim': int(
                    model_without_ddp.LOW_FEATURE_DIM
                ),
                'pivno_scale_gate': 'none',
            }
            observed_shared = dict(checkpoint_config)
            observed_shared['pivno_input_channels'] = (
                _infer_pivno_input_channels(checkpoint)
            )
            mismatches = {
                key: (observed_shared.get(key), expected)
                for key, expected in expected_shared.items()
                if observed_shared.get(key) != expected
            }
            if checkpoint_model == model_name:
                expected_sr = {
                    'model_variant': model_without_ddp.MODEL_VARIANT,
                    'pivno_mask_sr_base_model': (
                        model_without_ddp.BASE_MODEL_VARIANT
                    ),
                    'pivno_mask_sr_final_only': True,
                    'pivno_mask_sr_feature_channels': int(
                        model_without_ddp.sr_head.FEATURE_CHANNELS
                    ),
                    'pivno_mask_sr_input_channels': int(
                        model_without_ddp.sr_head.INPUT_CHANNELS
                    ),
                    'pivno_mask_sr_residual_max': float(
                        model_without_ddp.sr_head.residual_max
                    ),
                }
                mismatches.update({
                    key: (checkpoint_config.get(key), expected)
                    for key, expected in expected_sr.items()
                    if checkpoint_config.get(key) != expected
                })
            if mismatches:
                raise ValueError(
                    'GWC4/enc16/GRU3-mask-SR checkpoint/config mismatch: '
                    f'{mismatches}'
                )
            if checkpoint_model == base_model_name:
                expected_missing = {
                    key for key in model_without_ddp.state_dict()
                    if key.startswith('sr_head.')
                }
                if (
                    set(incompatible.missing_keys) != expected_missing
                    or incompatible.unexpected_keys
                ):
                    raise ValueError(
                        'GWC4/enc16/GRU3-mask-SR base initialization mismatch: '
                        f'missing={incompatible.missing_keys}, '
                        f'unexpected={incompatible.unexpected_keys}'
                    )
                logging.info(
                    'Loaded all GWC4/enc16/direct-concat/GRU3 base weights; '
                    'initialized only '
                    'the zero-output sr_head'
                )
            elif incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(
                    'GWC4/enc16/GRU3-mask-SR resume requires an exact state: '
                    f'missing={incompatible.missing_keys}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )
        if use_gated_gru3_sr:
            checkpoint_config = (
                checkpoint.get('model_config', {})
                if isinstance(checkpoint, dict) else {}
            )
            checkpoint_model = checkpoint_config.get('model')
            base_model_name = 'defom_pivno_gated_gru3'
            compatible_source_models = [base_model_name, model_name]
            if use_gated_gru3_mask_rgb_hidden_sr:
                compatible_source_models.append(
                    'defom_pivno_gated_gru3_gwc4_mask_sr'
                )
            if checkpoint_model not in compatible_source_models:
                raise ValueError(
                    f'{model_name} can initialize only from '
                    f'{compatible_source_models}, got {checkpoint_model!r}'
                )
            expected_shared = {
                'max_disp': int(args.max_disp),
                'n_downsample': int(args.n_downsample),
                'n_gru_layers': int(args.n_gru_layers),
                'hidden_dims': list(args.hidden_dims),
                'pivno_input_channels': 3,
                'pivno_scale_gate': model_without_ddp.SCALE_GATE_MODE,
                'corr_radius': int(args.corr_radius),
                'pivno_gru_kernel_size': int(
                    model_without_ddp.GRU_KERNEL_SIZE
                ),
                'pivno_right_sample_encoding': (
                    model_without_ddp.RIGHT_SAMPLE_ENCODING
                ),
                'pivno_match_num_groups': int(
                    model_without_ddp.MATCH_NUM_GROUPS
                ),
                'pivno_match_encoded_channels': int(
                    model_without_ddp.MATCH_ENCODED_CHANNELS
                ),
            }
            observed_shared = dict(checkpoint_config)
            observed_shared['pivno_input_channels'] = (
                _infer_pivno_input_channels(checkpoint)
            )
            mismatches = {
                key: (observed_shared.get(key), expected)
                for key, expected in expected_shared.items()
                if observed_shared.get(key) != expected
            }
            if checkpoint_model == model_name:
                expected_sr = {
                    'model_variant': model_without_ddp.MODEL_VARIANT,
                    'pivno_low_feature_dim': int(
                        model_without_ddp.LOW_FEATURE_DIM
                    ),
                    'pivno_mask_sr_base_model': (
                        model_without_ddp.BASE_MODEL_VARIANT
                    ),
                    'pivno_mask_sr_final_only': True,
                    'pivno_mask_sr_feature_channels': int(
                        model_without_ddp.sr_head.FEATURE_CHANNELS
                    ),
                    'pivno_mask_sr_input_channels': int(
                        model_without_ddp.sr_head.INPUT_CHANNELS
                    ),
                    'pivno_mask_sr_residual_max': float(
                        model_without_ddp.sr_head.residual_max
                    ),
                }
                if (
                    use_gated_gru3_mask_rgb_sr
                    or use_gated_gru3_mask_rgb_hidden_sr
                    or use_gated_gru3_mask_last_delta_sr
                    or use_gated_gru3_last_delta_direct_sr
                ):
                    expected_sr['pivno_mask_sr_feature_source'] = (
                        model_without_ddp.sr_head.FEATURE_SOURCE
                    )
                if use_gated_gru3_mask_last_delta_sr:
                    expected_sr.update({
                        'pivno_mask_sr_output': (
                            model_without_ddp.sr_head.OUTPUT_MODE
                        ),
                        'pivno_mask_sr_weight_mode': (
                            model_without_ddp.sr_head.WEIGHT_MODE
                        ),
                        'pivno_mask_sr_max_delta_disp_low': float(
                            model_without_ddp.sr_head.max_delta_disp_low
                        ),
                    })
                if use_gated_gru3_last_delta_direct_sr:
                    expected_sr.update({
                        'pivno_mask_sr_output': (
                            model_without_ddp.sr_head.OUTPUT_MODE
                        ),
                        'pivno_mask_sr_upsample_mode': (
                            model_without_ddp.sr_head.UPSAMPLE_MODE
                        ),
                        'pivno_mask_sr_max_delta_disp_low': float(
                            model_without_ddp.sr_head.max_delta_disp_low
                        ),
                        'pivno_mask_sr_max_delta_disp_hr': float(
                            model_without_ddp.sr_head.max_delta_disp_hr
                        ),
                        'pivno_mask_sr_final_composition': (
                            'previous_iteration_upsampled_disp_plus_direct_delta'
                        ),
                    })
                mismatches.update({
                    key: (checkpoint_config.get(key), expected)
                    for key, expected in expected_sr.items()
                    if checkpoint_config.get(key) != expected
                })
            if mismatches:
                raise ValueError(
                    'GWC4 gated-GRU3-mask-SR checkpoint/config mismatch: '
                    f'{mismatches}'
                )
            if checkpoint_model == base_model_name:
                expected_missing = {
                    key for key in model_without_ddp.state_dict()
                    if key.startswith('sr_head.')
                }
                if (
                    set(incompatible.missing_keys) != expected_missing
                    or incompatible.unexpected_keys
                ):
                    raise ValueError(
                        'GWC4 gated-GRU3-mask-SR base initialization mismatch: '
                        f'missing={incompatible.missing_keys}, '
                        f'unexpected={incompatible.unexpected_keys}'
                    )
                logging.info(
                    'Loaded all completed C32/GWC4 gated-GRU3 base weights; '
                    'initialized only the zero-output sr_head'
                )
            elif fusion_from_hidden_sr:
                expected_missing = {
                    key for key in model_without_ddp.state_dict()
                    if key.startswith('sr_head.image_encoder.')
                    or key.startswith('sr_head.feature_fusion.')
                }
                if (
                    set(incompatible.missing_keys) != expected_missing
                    or incompatible.unexpected_keys
                ):
                    raise ValueError(
                        'RGB/hidden fusion initialization did not preserve the '
                        'complete trained hidden-SR state: '
                        f'missing={incompatible.missing_keys}, '
                        f'unexpected={incompatible.unexpected_keys}'
                    )
                logging.info(
                    'Loaded the complete trained hidden-SR checkpoint; '
                    'initialized only identity/zero RGB fusion modules'
                )
            elif incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(
                    'GWC4 gated-GRU3-mask-SR resume requires an exact state: '
                    f'missing={incompatible.missing_keys}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )
        if use_gated_pivno:
            checkpoint_config = (
                checkpoint.get('model_config', {})
                if isinstance(checkpoint, dict) else {}
            )
            checkpoint_model = checkpoint_config.get('model')
            compatible_checkpoint_models = (
                (model_name,)
                if (
                    use_gated_gru1_pivno
                    or use_gated_gru3_pivno
                    or use_gated_gru3_gwc_only_pivno
                    or use_gated_gru_kernel_ablation
                )
                else ('defom_pivno', 'defom_pivno_gated')
            )
            if checkpoint_model not in compatible_checkpoint_models:
                raise ValueError(
                    f'{model_name} cannot initialize from '
                    f'{checkpoint_model!r}; expected one of '
                    f'{compatible_checkpoint_models}'
                )
            expected_shared = {
                'max_disp': int(args.max_disp),
                'n_downsample': int(args.n_downsample),
                'n_gru_layers': int(args.n_gru_layers),
                'hidden_dims': list(args.hidden_dims),
                'pivno_input_channels': 3,
            }
            if (
                use_gated_gru1_pivno
                or use_gated_gru3_pivno
                or use_gated_gru3_gwc_only_pivno
                or use_gated_gru_kernel_ablation
            ):
                expected_shared.update({
                    'pivno_gru_kernel_size': int(
                        model_without_ddp.GRU_KERNEL_SIZE
                    ),
                    'pivno_right_sample_encoding': (
                        model_without_ddp.RIGHT_SAMPLE_ENCODING
                    ),
                    'pivno_match_num_groups': int(
                        model_without_ddp.MATCH_NUM_GROUPS
                    ),
                    'pivno_match_encoded_channels': int(
                        model_without_ddp.MATCH_ENCODED_CHANNELS
                    ),
                })
                if use_gated_gru_kernel_ablation:
                    expected_shared['pivno_low_feature_dim'] = int(
                        model_without_ddp.LOW_FEATURE_DIM
                    )
            mismatches = {
                key: (checkpoint_config.get(key), expected)
                for key, expected in expected_shared.items()
                if checkpoint_config.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    'Gated PIVNO checkpoint/configuration mismatch: '
                    f'{mismatches}'
                )
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith('scale_gate.')
            ]
            if checkpoint_model == 'defom_pivno':
                expected_missing = {
                    key for key in model_without_ddp.state_dict()
                    if key.startswith('scale_gate.')
                }
                if set(incompatible.missing_keys) != expected_missing:
                    invalid_missing = list(incompatible.missing_keys)
            elif incompatible.missing_keys:
                invalid_missing = list(incompatible.missing_keys)
            if invalid_missing or incompatible.unexpected_keys:
                raise ValueError(
                    'Gated PIVNO checkpoint is incompatible: '
                    f'missing={incompatible.missing_keys}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )
            if checkpoint_model == 'defom_pivno':
                logging.info(
                    'Loaded all shared defom_pivno weights; initialized scale '
                    'gate uniformly'
                )
            else:
                logging.info('Resumed complete %s weights', model_name)
        if use_concat_gru3_pivno:
            checkpoint_config = (
                checkpoint.get('model_config', {})
                if isinstance(checkpoint, dict) else {}
            )
            checkpoint_model = checkpoint_config.get('model')
            compatible_checkpoint_models = (
                model_name,
                'defom_pivno_gated_gru3',
            )
            if checkpoint_model not in compatible_checkpoint_models:
                raise ValueError(
                    f'{model_name} cannot initialize from '
                    f'{checkpoint_model!r}; expected one of '
                    f'{compatible_checkpoint_models}'
                )
            expected_concat = {
                'state_mode': 'pivno_single_current_disp',
                'max_disp': int(args.max_disp),
                'n_downsample': int(args.n_downsample),
                'n_gru_layers': int(args.n_gru_layers),
                'hidden_dims': list(args.hidden_dims),
                'pivno_input_channels': 3,
                'pivno_gru_kernel_size': 3,
                'pivno_right_sample_encoding': (
                    model_without_ddp.RIGHT_SAMPLE_ENCODING
                ),
                'pivno_match_num_groups': int(
                    model_without_ddp.MATCH_NUM_GROUPS
                ),
                'pivno_match_encoded_channels': int(
                    model_without_ddp.MATCH_ENCODED_CHANNELS
                ),
            }
            if checkpoint_model == model_name:
                expected_concat.update({
                    'model_variant': model_without_ddp.MODEL_VARIANT,
                    'pivno_fusion_mode': model_without_ddp.FUSION_MODE,
                    'pivno_low_feature_dim': int(
                        model_without_ddp.LOW_FEATURE_DIM
                    ),
                    'pivno_scale_gate': 'none',
                })
            else:
                expected_concat['model_variant'] = (
                    'defom_pivno_gated_gru3'
                )
            mismatches = {
                key: (checkpoint_config.get(key), expected)
                for key, expected in expected_concat.items()
                if checkpoint_config.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    'Direct-concat PIVNO checkpoint/configuration mismatch: '
                    f'{mismatches}'
                )
            if checkpoint_model == model_name:
                valid_unexpected = set()
            else:
                valid_unexpected = {
                    key for key in checkpoint_state
                    if key.startswith('scale_gate.')
                }
                expected_gate_keys = {
                    'scale_gate.0.weight',
                    'scale_gate.0.bias',
                    'scale_gate.2.weight',
                    'scale_gate.2.bias',
                }
                if valid_unexpected != expected_gate_keys:
                    raise ValueError(
                        'Gated GRU3 initialization has an unexpected scale '
                        f'gate state: {sorted(valid_unexpected)}'
                    )
            if (
                incompatible.missing_keys
                or set(incompatible.unexpected_keys) != valid_unexpected
            ):
                raise ValueError(
                    'Direct-concat PIVNO checkpoint state is incompatible: '
                    f'missing={incompatible.missing_keys}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )
            if checkpoint_model == model_name:
                logging.info('Resumed complete %s weights', model_name)
            else:
                logging.info(
                    'Loaded all shared GWC4/enc16 GRU3 weights; discarded '
                    'only scale_gate parameters'
                )
        if use_pact_bilap and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise ValueError(f'PACT-BiLap-GRU resume requires an exact model state: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}')
        if use_pact_smd and not args.strict_resume:
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith('d0_smd_head.')
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise ValueError(
                    'PACT-SMD accepts only a missing d0_smd_head when loading '
                    'a base PACT checkpoint; '
                    f'missing={invalid_missing}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )
            if incompatible.missing_keys:
                logging.info(
                    'Initialized the new PACT-SMD head; loaded all shared PACT weights'
                )
        if use_pact_smd_post:
            if post_source_model == 'pact_smd':
                invalid_missing = [
                    key for key in incompatible.missing_keys
                    if not key.startswith('final_smd_head.')
                ]
                if invalid_missing or incompatible.unexpected_keys:
                    raise ValueError(
                        'PACT-SMD-post Full initialization accepts only a '
                        'missing final_smd_head; '
                        f'missing={invalid_missing}, '
                        f'unexpected={incompatible.unexpected_keys}'
                    )
                expected_new = {
                    name for name in model_without_ddp.state_dict()
                    if name.startswith('final_smd_head.')
                }
                if set(incompatible.missing_keys) != expected_new:
                    raise ValueError(
                        'PACT-SMD-post did not observe the complete new-head '
                        f'key set: missing={incompatible.missing_keys}'
                    )
                logging.info(
                    'Loaded the complete frozen Full model; initialized only '
                    'final_smd_head'
                )
            elif incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(
                    'PACT-SMD-post resume requires an exact state: '
                    f'missing={incompatible.missing_keys}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )

        checkpoint_model_name = (
            checkpoint.get('model_config', {}).get('model')
            if isinstance(checkpoint, dict) else None
        )
        same_optimizer_architecture = (
            not (use_pact_smd or use_pact_smd_post)
            or (
                use_pact_smd
                and
                checkpoint_model_name == 'pact_smd'
                and checkpoint.get('model_config', {}).get('pact_smd_stage')
                == args.pact_smd_stage
            )
            or (
                use_pact_smd_post
                and checkpoint_model_name == 'pact_smd_post'
            )
        ) and (
            not (
                use_gated_pivno
                or use_concat_gru3_family
                or use_gated_gru3_sr
            )
            or checkpoint_model_name == model_name
        ) and (
            not use_mask_sr
            or checkpoint_model_name == model_name
        )
        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer and same_optimizer_architecture:
            print('Load optimizer')
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            resume_batch_in_epoch = checkpoint.get('batch_in_epoch')
            resume_scaler_state = checkpoint.get('scaler')
            del optimizer, scheduler
            optimizer, scheduler = fetch_optimizer(args, model, start_step, checkpoint)
        del checkpoint

    train_data = datasets.fetch_dataset(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=args.num_workers > 0,
    )

    total_steps = start_step
    benchmark_times = []
    epoch = start_epoch
    logger = Logger(model, scheduler, args.name) if is_main_process else None
    if logger is not None:
        logger.total_steps = total_steps

    model.train()
    if use_pact:
        # model.train() re-enables BatchNorm, so freeze it afterwards.
        model_without_ddp.freeze_bn()
    # The from-scratch PACT auxiliary losses are initially much larger than
    # the mature recurrent loss. A conservative scale avoids several skipped
    # optimizer steps while retaining dynamic growth.
    scaler = GradScaler(
        enabled=args.mixed_precision,
        init_scale=128.0 if use_pact else 65536.0,
    )
    if resume_scaler_state is not None and args.mixed_precision:
        scaler.load_state_dict(resume_scaler_state)

    debug_finite_steps = max(
        0, int(getattr(args, 'pact_debug_finite_steps', 0))
    )
    debug_finite_end_step = start_step + debug_finite_steps
    if use_pact:
        model_without_ddp.debug_finite = debug_finite_steps > 0
    if is_main_process and debug_finite_steps > 0:
        logging.info(
            "PACT finite checks enabled for steps [%d, %d)",
            start_step,
            debug_finite_end_step,
        )
    should_keep_training = True
    reported_grad_stride_mismatch = False

    while should_keep_training:

        # mannually change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and total_steps == start_step:
            if resume_batch_in_epoch is None:
                epoch_start_step = start_step - len(train_loader)*start_epoch
            else:
                epoch_start_step = int(resume_batch_in_epoch)
        else:
            epoch_start_step = 0

        if not 0 <= epoch_start_step <= len(train_loader):
            raise ValueError(
                "invalid checkpoint batch position: "
                f"epoch={epoch}, step={total_steps}, "
                f"batch_in_epoch={epoch_start_step}, "
                f"loader_length={len(train_loader)}"
            )
        inferred_batch_in_epoch = start_step - len(train_loader) * start_epoch
        if (
            epoch == start_epoch
            and total_steps == start_step
            and resume_batch_in_epoch is not None
            and epoch_start_step != inferred_batch_in_epoch
        ):
            raise ValueError(
                "checkpoint batch position is inconsistent with the current "
                "loader: "
                f"stored={epoch_start_step}, inferred={inferred_batch_in_epoch}, "
                f"epoch={start_epoch}, step={start_step}, "
                f"loader_length={len(train_loader)}"
            )

        epoch_loader = train_loader
        if epoch_start_step > 0:
            if is_main_process:
                logging.info(
                    "Fast-resuming epoch %d: skipping %d/%d completed batch "
                    "indices without loading data",
                    epoch, epoch_start_step, len(train_loader),
                )
            epoch_loader = DataLoader(
                dataset=train_data,
                batch_sampler=OffsetBatchSampler(
                    train_loader.batch_sampler, epoch_start_step
                ),
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )

        progress = tqdm(
            epoch_loader,
            total=len(train_loader),
            initial=epoch_start_step,
            disable=not is_main_process,
        )
        for i_batch, data_blob in enumerate(
            progress, start=epoch_start_step
        ):
            if benchmark_mode:
                torch.cuda.synchronize(device)
                step_started = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            image1 = data_blob["img1"].to(device, non_blocking=True)
            image2 = data_blob["img2"].to(device, non_blocking=True)
            disp_gt = data_blob["disp"].to(device, non_blocking=True)
            valid = data_blob["valid"].to(device, non_blocking=True)

            debug_finite_active = (
                use_pact
                and debug_finite_steps > 0
                and total_steps < debug_finite_end_step
            )
            if use_pact:
                model_without_ddp.debug_finite = debug_finite_active

            assert model.training
            if use_pact:
                disp_predictions, pact_aux = model(
                    image1,
                    image2,
                    iters=args.train_iters,
                    scale_iters=args.scale_iters,
                    return_aux=True,
                )
            elif use_pivno:
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    pivno_init_predictions, pivno_recurrent_predictions = model(
                        image1,
                        image2,
                        iters=args.train_iters,
                        scale_iters=args.scale_iters,
                    )
                if not pivno_init_predictions:
                    raise RuntimeError("PACT-PIVNO returned no initialization predictions")
                disp_predictions = [
                    *pivno_init_predictions,
                    *pivno_recurrent_predictions,
                ]
                pact_aux = None
            else:
                disp_predictions = model(
                    image1,
                    image2,
                    iters=args.train_iters,
                    scale_iters=args.scale_iters,
                )
                pact_aux = None
            assert model.training

            try:
                if use_pivno:
                    sequence_loss, metrics = pact_pivno_sequence_loss(
                        pivno_init_predictions,
                        pivno_recurrent_predictions,
                        disp_gt,
                        valid,
                        max_disp=args.max_disp,
                    )
                    if use_scale_gate_pivno:
                        metrics.update(model_without_ddp.scale_gate_metrics())
                elif use_pact_bilap:
                    sequence_loss, metrics = bilap_sequence_loss(disp_predictions, disp_gt, valid, max_disp=args.max_disp, gamma=0.9, nll_weight=args.bilap_nll_weight, map_weight=args.bilap_map_weight, edge_weight=args.bilap_edge_weight, diversity_weight=args.bilap_diversity_weight, diversity_margin=args.bilap_diversity_margin, nll_edge_only=args.bilap_nll_region == 'edge')
                else:
                    sequence_loss, metrics = sequence_loss_d0L1_edge(
                        disp_predictions,
                        disp_gt,
                        valid,
                        max_flow=args.max_disp,
                        delta_info_preds=(
                            pact_aux.get("delta_info_preds")
                            if use_pact and not use_pact2
                            and not use_pact_smd_post else None
                        ),
                    )
                    if use_pact2:
                        metrics.pop('mol_confidence_mean', None)
                        metrics.pop('mol_log_b_mean', None)
            except FloatingPointError:
                logging.exception(
                    "Non-finite disparity at rank=%d step=%d epoch=%d "
                    "batch=%d imageL=%s disp=%s",
                    rank,
                    total_steps,
                    epoch,
                    i_batch,
                    list(data_blob.get("imageL_file", [])),
                    list(data_blob.get("disp_file", [])),
                )
                raise
            if use_pact_bilap:
                base_loss, base_metrics = pact_auxiliary_loss(pact_aux['base_aux'], disp_gt, valid, max_disp=args.max_disp)
                smd_loss, smd_metrics = pact_smd_auxiliary_loss(pact_aux, disp_gt, valid, max_disp=args.max_disp, stage='joint', nll_weight=args.pact_smd_nll_joint_weight, selection_weight=args.pact_smd_selection_weight, guard_weight=args.pact_smd_guard_weight, nll_edge_only=args.bilap_nll_region == 'edge')
                auxiliary_loss = base_loss + smd_loss
                auxiliary_metrics = {**base_metrics, **smd_metrics}
                sequence_weight = 1.0
            elif use_pact_smd_post:
                post_loss, post_metrics = pact_smd_post_auxiliary_loss(
                    pact_aux,
                    disp_gt,
                    valid,
                    max_disp=args.max_disp,
                    nll_weight=args.pact_smd_post_nll_weight,
                    selection_weight=args.pact_smd_post_selection_weight,
                    guard_weight=args.pact_smd_post_guard_weight,
                    bad3_weight=args.pact_smd_post_bad3_weight,
                    edge_weight=args.pact_smd_post_edge_weight,
                    guard_margin=args.pact_smd_post_guard_margin,
                    bad3_tau=args.pact_smd_post_bad3_tau,
                )
                auxiliary_loss = post_loss
                auxiliary_metrics = post_metrics
                sequence_weight = args.pact_smd_post_map_weight
            elif use_pact_smd:
                nll_weight = (
                    args.pact_smd_nll_head_weight
                    if args.pact_smd_stage == 'head'
                    else args.pact_smd_nll_full_weight
                    if args.pact_smd_stage == 'full'
                    else args.pact_smd_nll_joint_weight
                )
                smd_loss, smd_metrics = pact_smd_auxiliary_loss(
                    pact_aux,
                    disp_gt,
                    valid,
                    max_disp=args.max_disp,
                    stage=args.pact_smd_stage,
                    nll_weight=nll_weight,
                    selection_weight=args.pact_smd_selection_weight,
                    guard_weight=args.pact_smd_guard_weight,
                )
                if args.pact_smd_stage == 'full':
                    # Supervise the original PACT initialization directly so
                    # the randomly initialized coarse/initializer path cannot
                    # rely on the new head to hide a weak d0_old.
                    base_aux = dict(pact_aux)
                    base_aux['init_disp'] = pact_aux['init_disp_old']
                    base_loss, base_metrics = pact_auxiliary_loss(
                        base_aux,
                        disp_gt,
                        valid,
                        max_disp=args.max_disp,
                    )
                    auxiliary_loss = (
                        float(args.pact_smd_base_aux_weight) * base_loss
                        + smd_loss
                    )
                    auxiliary_metrics = {**base_metrics, **smd_metrics}
                    auxiliary_metrics['smd_base_aux_weight'] = float(
                        args.pact_smd_base_aux_weight
                    )
                else:
                    auxiliary_loss = smd_loss
                    auxiliary_metrics = smd_metrics
                sequence_weight = (
                    args.pact_smd_head_sequence_weight
                    if args.pact_smd_stage == 'head' else 1.0
                )
            elif use_pact:
                auxiliary_loss, auxiliary_metrics = pact_auxiliary_loss(
                    pact_aux,
                    disp_gt,
                    valid,
                    max_disp=args.max_disp,
                )
                sequence_weight = 1.0
            else:
                auxiliary_loss = sequence_loss.new_zeros(())
                auxiliary_metrics = {}
                sequence_weight = 1.0
            loss = float(sequence_weight) * sequence_loss + auxiliary_loss
            metrics.update(auxiliary_metrics)
            metrics['sequence_objective'] = sequence_loss.detach().item()
            metrics['sequence_weight'] = float(sequence_weight)
            metrics['total_loss'] = loss.detach().item()
            if use_pact_bilap:
                # Keep the BiLap console/TensorBoard summary directly
                # comparable to the legacy PACT summary. PACT reports the
                # fractions below each pixel threshold, rather than outlier
                # fractions. Detailed loss terms still participate in
                # ``loss`` above; they are only omitted from this summary.
                final_epe = metrics['bilap_final_epe']
                final_prediction = disp_predictions[-1]['disp'].float()
                target = disp_gt.unsqueeze(1) if disp_gt.ndim == 3 else disp_gt
                valid_mask = valid.unsqueeze(1) if valid.ndim == 3 else valid
                valid_mask = (valid_mask >= 0.5) & torch.isfinite(target) & (target >= 0.0) & (target < float(args.max_disp))
                absolute_error = (final_prediction - target.float()).abs()
                valid_error = absolute_error[valid_mask]
                metrics = {
                    'epe': final_epe,
                    '1px': (valid_error < 1.0).float().mean().item(),
                    '3px': (valid_error < 3.0).float().mean().item(),
                    '5px': (valid_error < 5.0).float().mean().item(),
                }

            if debug_finite_active and not bool(
                torch.isfinite(loss.detach()).item()
            ):
                raise FloatingPointError(
                    "PACT non-finite loss before backward: "
                    f"step={total_steps}, sequence={sequence_loss.detach().item()}, "
                    f"auxiliary={auxiliary_loss.detach().item()}, "
                    f"total={loss.detach().item()}"
                )

            scaler.scale(loss).backward()

            if (
                not reported_grad_stride_mismatch
                and args.distributed
                and is_main_process
            ):
                mismatch_lines = []
                for name, param in model_without_ddp.named_parameters():
                    if param.grad is None:
                        continue
                    if tuple(param.grad.stride()) != tuple(param.stride()):
                        mismatch_lines.append(
                            f"{name}: grad_shape={tuple(param.grad.shape)}, "
                            f"grad_stride={tuple(param.grad.stride())}, "
                            f"param_stride={tuple(param.stride())}"
                        )
                if mismatch_lines:
                    logging.warning("DDP grad/param stride mismatch detected:\n%s", "\n".join(mismatch_lines[:20]))
                    reported_grad_stride_mismatch = True

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_finite = bool(torch.isfinite(grad_norm.detach()).item())
            if debug_finite_active and not grad_norm_finite:
                if args.mixed_precision:
                    if is_main_process:
                        logging.warning(
                            "PACT AMP gradient overflow at step=%d, "
                            "grad_norm=%s, scale=%g; GradScaler will skip "
                            "this optimizer step and reduce the scale",
                            total_steps,
                            grad_norm.detach().item(),
                            scaler.get_scale(),
                        )
                else:
                    raise FloatingPointError(
                        "PACT non-finite FP32 gradient norm before optimizer "
                        f"step: step={total_steps}, "
                        f"grad_norm={grad_norm.detach().item()}"
                    )
            if debug_finite_active:
                metrics['amp_grad_overflow'] = float(not grad_norm_finite)

            # With AMP, optimizer.step() may be skipped on overflow.
            # Step LR scheduler only when the optimizer really updates.
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            total_steps += 1

            if benchmark_mode:
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - step_started, device=device)
                if args.distributed:
                    torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                elapsed_seconds = elapsed.item()
                if total_steps > args.benchmark_warmup:
                    benchmark_times.append(elapsed_seconds)
                if is_main_process:
                    measured_avg = (
                        sum(benchmark_times) / len(benchmark_times)
                        if benchmark_times else float('nan')
                    )
                    print(
                        f"BENCHMARK step={total_steps}/{args.benchmark_steps} "
                        f"slowest_rank={elapsed_seconds:.3f}s "
                        f"measured_avg={measured_avg:.3f}s",
                        flush=True,
                    )

            run_validation = (not benchmark_mode) and total_steps % args.val_freq == 0

            if is_main_process:
                logger.writer.add_scalar("train/live_loss", loss.item(), total_steps)
                logger.writer.add_scalar(f'train/learning_rate', optimizer.param_groups[0]['lr'], total_steps)
                logger.push(metrics)

                if (not benchmark_mode) and total_steps % args.save_latest_ckpt_freq == 0:
                    save_path = Path('checkpoints/%s/checkpoint_latest.pth' % (args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    save_dict = { 'model': model_without_ddp.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'scheduler': scheduler.state_dict(),
                                  'scaler': scaler.state_dict(),
                                  'step': total_steps,
                                  'epoch': epoch,
                                  'batch_in_epoch': i_batch + 1,
                                  'model_config': model_config}
                    torch.save(save_dict, save_path)

                if (not benchmark_mode) and total_steps % args.save_ckpt_freq == 0:
                    save_path = Path('checkpoints/%s/%s_%6d.pth' % (args.name, args.name, total_steps))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'model_config': model_config,
                        'step': total_steps,
                    }, save_path)

                if run_validation:
                    # visualizing training results with tensorboard
                    disp = disp_predictions[-1]['disp'] if use_pact_bilap else disp_predictions[-1]

                    for j in range(min(4, args.batch_size)):  # write a maxmimum of four images
                        logger.writer.add_image("image1/{}".format(j), image1[j].data.type(torch.uint8), total_steps)
                        logger.writer.add_image("image2/{}".format(j), image2[j].data.type(torch.uint8), total_steps)
                        logger.writer.add_image("disp/{}".format(j),
                                                (disp[j]).data.type(torch.uint8), total_steps)
                        logger.writer.add_image("gt_disp/{}".format(j),
                                                (disp_gt[j]).data.type(torch.uint8), total_steps)

            if run_validation:
                if args.distributed:
                    torch.distributed.barrier()
                # validate_things partitions the dataset by rank and uses an
                # all_reduce to aggregate its metrics. Every distributed rank
                # must therefore enter validation; otherwise rank 0 waits in
                # the metric all_reduce while the remaining ranks wait at the
                # barrier below.
                results = validate_things(
                    model_without_ddp,
                    args.valid_iters,
                    args.scale_iters,
                    mixed_prec=args.mixed_precision,
                    max_disp=validation_max_disp,
                )
                if is_main_process:
                    logger.write_dict(results)
                if args.distributed:
                    torch.distributed.barrier()
                model.train()
                if use_pact:
                    model_without_ddp.freeze_bn()

            if total_steps >= args.num_steps:
                should_keep_training = False
                break

        epoch += 1

        if (
            (not benchmark_mode)
            and is_main_process
            and args.save_epoch_checkpoint
            and len(train_loader) >= 10000
        ):
            save_path = Path('checkpoints/%s/%d_epoch_%s.pth.gz' % (args.name, total_steps, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save({
                'model': model_without_ddp.state_dict(),
                'model_config': model_config,
                'step': total_steps,
            }, save_path)

    PATH = None if benchmark_mode else 'checkpoints/%s.pth' % args.name
    if benchmark_mode and device.type == 'cuda':
        peak_memory = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], dtype=torch.float64, device=device)
        if args.distributed:
            torch.distributed.all_reduce(peak_memory, op=torch.distributed.ReduceOp.MAX)
        benchmark_peak_memory = peak_memory.cpu().tolist()
    if args.distributed:
        torch.distributed.barrier()
    if is_main_process:
        if benchmark_mode and benchmark_times:
            memory_suffix = "" if benchmark_peak_memory is None else (f" peak_allocated={benchmark_peak_memory[0] / (1024 ** 3):.3f}GiB" f" peak_reserved={benchmark_peak_memory[1] / (1024 ** 3):.3f}GiB")
            print(
                f"BENCHMARK RESULT: gpus={world_size} "
                f"global_batch={global_batch_size} warmup={args.benchmark_warmup} "
                f"measured_steps={len(benchmark_times)} "
                f"avg_slowest_rank={sum(benchmark_times) / len(benchmark_times):.3f}s "
                f"min={min(benchmark_times):.3f}s max={max(benchmark_times):.3f}s"
                f"{memory_suffix}",
                flush=True,
            )
        elif benchmark_mode:
            print(
                "BENCHMARK RESULT: no measured steps; benchmark_steps must "
                "exceed benchmark_warmup",
                flush=True,
            )
        else:
            print("FINISHED TRAINING")
        logger.close()
        if not benchmark_mode and args.save_final_checkpoint:
            torch.save({
                'model': model_without_ddp.state_dict(),
                'model_config': model_config,
                'step': total_steps,
            }, PATH)
    if args.distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='defom-stereo', help="name your experiment")
    parser.add_argument(
        '--model',
        choices=['legacy', 'pact_pivno', 'defom_pivno', 'defom_pivno_mobilenetv2', 'defom_pivno_gated', 'defom_pivno_gated_gru1', 'defom_pivno_gated_gru3', 'defom_pivno_gated_gru3_gwc_only', 'defom_pivno_gated_gru_kernel_ablation', 'defom_pivno_gated_gru3_gwc4_mask_sr', 'defom_pivno_gated_gru3_gwc4_mask_rgb_sr', 'defom_pivno_gwc4_enc16_concat_gru3', 'defom_pivno_gwc4_enc16_concat_gru3_mask_sr'],
        default='legacy',
        help="model family; PACT is opt-in to preserve existing checkpoints",
    )

    # resume pretrained model or resume training
    parser.add_argument('--resume_ckpt', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--train_folds', type=int, nargs='+', default=[1], help="training datasets' folds.")
    parser.add_argument('--lr', type=float, default=0.00002, help="max learning rate.")
    parser.add_argument(
        '--pivno_gate_lr',
        type=float,
        default=None,
        help='max LR for defom_pivno_gated scale_gate; defaults to --lr',
    )
    parser.add_argument(
        '--pivno_gru_kernel_size',
        type=int,
        choices=[1, 3],
        default=None,
        help=(
            'outer ConvGRU kernel for '
            'defom_pivno_gated_gru_kernel_ablation; must be 1 or 3'
        ),
    )
    parser.add_argument(
        '--pivno_mask_sr_stage',
        choices=['head', 'joint'],
        default='head',
        help='train only the final mask-guided SR head or all model weights',
    )
    parser.add_argument(
        '--pivno_mask_sr_residual_max',
        type=float,
        default=4.0,
        help='absolute full-resolution pixel bound for SR delta_d',
    )
    parser.add_argument('--image_size', type=int, nargs='+', default=[352, 768], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=18, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--seed', default=1234565, type=int)

    # log
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--benchmark_steps', type=int, default=0,
                        help='run only N synchronized training steps without validation/checkpoints')
    parser.add_argument('--benchmark_warmup', type=int, default=3,
                        help='initial benchmark steps excluded from the timing summary')
    parser.add_argument('--save_ckpt_freq', default=10000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--save_epoch_checkpoint', action=argparse.BooleanOptionalAction, default=True, help='save a model-only checkpoint at every completed training epoch')
    parser.add_argument('--save_final_checkpoint', action=argparse.BooleanOptionalAction, default=True, help='save checkpoints/<name>.pth after training completes')
    parser.add_argument('--val_freq', default=10000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--max_disp', type=int, default=768, help="maximum disparity")
    parser.add_argument(
        '--eval_max_disp', type=float, default=None,
        help=(
            'independent upper bound for valid GT pixels during validation; '
            'defaults to --max_disp and <=0 disables the bound'
        ),
    )
    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disparity field updates during validation forward pass')

    # Raft Architecure choices
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--corr_levels', type=int, default=3, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")

    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2, help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0.0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default='v', choices=['v', 'None'], help='flip the images vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    Path("checkpoints/"+args.name).mkdir(exist_ok=True, parents=True)

    train(args)
