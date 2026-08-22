#!/usr/bin/env python3
"""Evaluate the final full-resolution PIVNO initialization (d0) on SceneFlow."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIVNO.models.sronet import PIVNO  # noqa: E402
from core.stereo_datasets import SceneFlowDatasets  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


PIVNO_PREFIX = "pivno."
PIVNO_INPUT_WEIGHT = "pivno.snet.conv1.weight"
SEARCH_SUPPORT_THRESHOLDS = (16.0, 32.0, 64.0)


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return dict(state)


def _checkpoint_state(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint['model'] must be a state-dict mapping")
    return _strip_module_prefix(state)


def infer_pivno_input_channels(state: Mapping[str, torch.Tensor]) -> int:
    if PIVNO_INPUT_WEIGHT not in state:
        raise KeyError(f"checkpoint is missing {PIVNO_INPUT_WEIGHT!r}")
    weight = state[PIVNO_INPUT_WEIGHT]
    if weight.ndim != 4:
        raise ValueError(
            f"{PIVNO_INPUT_WEIGHT} must be a Conv2d weight, got {tuple(weight.shape)}"
        )
    channels = int(weight.shape[1])
    if channels not in (1, 3):
        raise ValueError(f"unsupported PIVNO input channel count: {channels}")
    return channels


def extract_pivno_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    pivno_state = {
        key[len(PIVNO_PREFIX):]: value
        for key, value in state.items()
        if key.startswith(PIVNO_PREFIX)
    }
    if not pivno_state:
        raise KeyError("checkpoint contains no 'pivno.*' parameters")
    return pivno_state


def prepare_pivno_input(image: torch.Tensor, input_channels: int) -> torch.Tensor:
    """Reproduce the RGB or legacy grayscale PIVNO training input."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected RGB input [B,3,H,W], got {tuple(image.shape)}")
    if input_channels == 3:
        return image / 255.0
    if input_channels == 1:
        coefficients = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * coefficients).sum(dim=1, keepdim=True) / 255.0
    raise ValueError(f"unsupported PIVNO input channel count: {input_channels}")


def _distributed_info(device_name: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed:
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0
    return distributed, rank, world_size, device


def _reduce_sums(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu()


def _batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@torch.no_grad()
def evaluate(args) -> Dict[str, object]:
    distributed, rank, world_size, device = _distributed_info(args.device)
    is_main = rank == 0
    try:
        checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
        state = _checkpoint_state(checkpoint)
        input_channels = infer_pivno_input_channels(state)
        model = PIVNO(input_channels=input_channels)
        model.load_state_dict(extract_pivno_state(state), strict=True)
        model.to(device).eval()

        config = checkpoint.get("model_config", {}) if isinstance(checkpoint, Mapping) else {}
        checkpoint_max_disp = config.get("max_disp") if isinstance(config, Mapping) else None
        max_disp = float(
            args.max_disp
            if args.max_disp is not None
            else checkpoint_max_disp if checkpoint_max_disp is not None
            else 768.0
        )

        dataset = SceneFlowDatasets(
            root=args.sceneflow_root,
            dstype="frames_finalpass",
            things_test=True,
        )
        if len(dataset) == 0:
            raise RuntimeError(
                f"no FlyingThings3D TEST samples found under {args.sceneflow_root}"
            )
        indices = list(range(rank, len(dataset), world_size))

        if is_main:
            print(
                f"checkpoint={args.restore_ckpt} step={checkpoint.get('step', 'unknown')} "
                f"pivno_input_channels={input_channels} max_disp={max_disp:g} "
                f"images={len(dataset)} world_size={world_size} batch_per_gpu={args.batch_size}"
            )

        image_epe_sum = 0.0
        absolute_error_sum = 0.0
        valid_pixel_count = 0
        image_count = 0
        skipped_image_count = 0
        bad_counts = {1.0: 0, 3.0: 0, 5.0: 0}
        image_bad_rate_sums = {1.0: 0.0, 3.0: 0.0, 5.0: 0.0}
        accurate_counts = {1.0: 0, 3.0: 0, 5.0: 0}
        support_counts = {
            threshold: 0 for threshold in SEARCH_SUPPORT_THRESHOLDS
        }

        progress = tqdm(
            list(_batched(indices, args.batch_size)),
            disable=not is_main,
            desc="PIVNO d0",
        )
        for batch_ids in progress:
            samples = [dataset[index] for index in batch_ids]
            shapes = {tuple(sample["img1"].shape) for sample in samples}
            if len(shapes) != 1:
                raise ValueError(
                    f"batch {batch_ids} contains unequal image shapes: {sorted(shapes)}"
                )
            image1 = torch.stack([sample["img1"] for sample in samples]).to(device)
            image2 = torch.stack([sample["img2"] for sample in samples]).to(device)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            pivno_image1 = prepare_pivno_input(image1, input_channels).contiguous().float()
            pivno_image2 = prepare_pivno_input(image2, input_channels).contiguous().float()

            with torch.cuda.amp.autocast(
                enabled=args.mixed_precision and device.type == "cuda"
            ):
                # PIVNO emits five full-resolution initializations. The last
                # one is generated from the same quarter-resolution tensor
                # used as d0 by the recurrent DEFOM refinement.
                d0_prediction = model(pivno_image1, pivno_image2)[-1]
            d0_prediction = padder.unpad(d0_prediction).float().cpu()

            for batch_index, sample in enumerate(samples):
                prediction = d0_prediction[batch_index]
                ground_truth = sample["disp"].float()
                valid = (
                    (sample["valid"] >= 0.5)
                    & torch.isfinite(ground_truth)
                    & (ground_truth >= 0.0)
                    & (ground_truth < max_disp)
                )
                if not bool(valid.any()):
                    skipped_image_count += 1
                    continue
                if prediction.shape != ground_truth.shape:
                    raise ValueError(
                        f"prediction/GT shape mismatch for {sample['imageL_file']}: "
                        f"{tuple(prediction.shape)} vs {tuple(ground_truth.shape)}"
                    )
                if not bool(torch.isfinite(prediction[valid]).all()):
                    raise FloatingPointError(
                        f"non-finite d0 prediction for {sample['imageL_file']}"
                    )

                error = (prediction - ground_truth).abs()[valid]
                image_epe_sum += float(error.mean())
                absolute_error_sum += float(error.sum())
                valid_pixel_count += int(error.numel())
                image_count += 1
                for threshold in bad_counts:
                    bad_count = int((error > threshold).sum())
                    bad_counts[threshold] += bad_count
                    image_bad_rate_sums[threshold] += bad_count / error.numel()
                    accurate_counts[threshold] += int((error < threshold).sum())
                for threshold in support_counts:
                    support_counts[threshold] += int((error <= threshold).sum())

        reduced = _reduce_sums(
            [
                image_epe_sum,
                absolute_error_sum,
                valid_pixel_count,
                image_count,
                skipped_image_count,
                bad_counts[1.0],
                bad_counts[3.0],
                bad_counts[5.0],
                accurate_counts[1.0],
                accurate_counts[3.0],
                accurate_counts[5.0],
                image_bad_rate_sums[1.0],
                image_bad_rate_sums[3.0],
                image_bad_rate_sums[5.0],
                *[support_counts[value] for value in SEARCH_SUPPORT_THRESHOLDS],
            ],
            device,
        ).tolist()
        (
            image_epe_sum,
            absolute_error_sum,
            valid_pixel_count,
            image_count,
            skipped_image_count,
            bad1,
            bad3,
            bad5,
            acc1,
            acc3,
            acc5,
            image_bad1_sum,
            image_bad3_sum,
            image_bad5_sum,
        ) = reduced[:14]
        reduced_support_counts = dict(zip(SEARCH_SUPPORT_THRESHOLDS, reduced[14:]))
        if image_count <= 0 or valid_pixel_count <= 0:
            raise RuntimeError("evaluation produced no valid images or pixels")

        results = {
            "checkpoint": str(Path(args.restore_ckpt).resolve()),
            "checkpoint_step": checkpoint.get("step"),
            "dataset": "FlyingThings3D TEST frames_finalpass",
            "dataset_root": str(Path(args.sceneflow_root).resolve()),
            "d0_definition": "PIVNO init_disp_predictions[-1] (full resolution)",
            "pivno_input_channels": input_channels,
            "max_disp": max_disp,
            "images": int(image_count),
            "skipped_images": int(skipped_image_count),
            "valid_pixels": int(valid_pixel_count),
            "d0_epe_image_mean": image_epe_sum / image_count,
            "d0_epe_pixel_mean": absolute_error_sum / valid_pixel_count,
            "d0_bad1_image_mean_percent": 100.0 * image_bad1_sum / image_count,
            "d0_bad3_image_mean_percent": 100.0 * image_bad3_sum / image_count,
            "d0_bad5_image_mean_percent": 100.0 * image_bad5_sum / image_count,
            "d0_bad1_pixel_mean_percent": 100.0 * bad1 / valid_pixel_count,
            "d0_bad3_pixel_mean_percent": 100.0 * bad3 / valid_pixel_count,
            "d0_bad5_pixel_mean_percent": 100.0 * bad5 / valid_pixel_count,
            "d0_acc1_pixel_mean_percent": 100.0 * acc1 / valid_pixel_count,
            "d0_acc3_pixel_mean_percent": 100.0 * acc3 / valid_pixel_count,
            "d0_acc5_pixel_mean_percent": 100.0 * acc5 / valid_pixel_count,
        }
        previous_threshold = 0.0
        previous_count = 0.0
        for threshold in SEARCH_SUPPORT_THRESHOLDS:
            threshold_count = reduced_support_counts[threshold]
            label = f"{threshold:g}"
            previous_label = f"{previous_threshold:g}"
            results[f"d0_within_{label}px_percent"] = (
                100.0 * threshold_count / valid_pixel_count
            )
            results[f"d0_error_{previous_label}_{label}px_percent"] = (
                100.0 * (threshold_count - previous_count) / valid_pixel_count
            )
            previous_threshold = threshold
            previous_count = threshold_count
        largest_label = f"{SEARCH_SUPPORT_THRESHOLDS[-1]:g}"
        results[f"d0_error_over_{largest_label}px_percent"] = (
            100.0 * (valid_pixel_count - previous_count) / valid_pixel_count
        )

        if is_main:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(f"saved={output_path}")
        return results
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_ckpt", required=True)
    parser.add_argument(
        "--sceneflow_root",
        default=os.environ.get("SCENEFLOW_ROOT", "/data/public_data"),
    )
    parser.add_argument(
        "--max_disp",
        type=float,
        default=None,
        help="valid GT upper bound; defaults to checkpoint model_config.max_disp",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch per GPU")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output_json")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(parse_args())
