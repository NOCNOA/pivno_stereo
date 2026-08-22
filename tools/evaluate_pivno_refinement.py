#!/usr/bin/env python3
"""Evaluate how recurrent DEFOM refinement changes PIVNO d0 errors."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pivno_models.defom_pivno import DEFOMStereo  # noqa: E402
from core.stereo_datasets import SceneFlowDatasets  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402
from utils.utils import disparity_edge_mask  # noqa: E402


D0_ERROR_THRESHOLDS = (16.0, 32.0, 64.0)


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


def build_d0_error_masks(
    d0_error: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Sequence[float] = D0_ERROR_THRESHOLDS,
) -> Dict[str, torch.Tensor]:
    """Create mutually exclusive masks from the full-resolution d0 error."""
    masks: Dict[str, torch.Tensor] = {}
    previous = 0.0
    for index, threshold in enumerate(thresholds):
        lower = d0_error >= previous if index == 0 else d0_error > previous
        label = f"{previous:g}_{threshold:g}px"
        masks[label] = valid & lower & (d0_error <= threshold)
        previous = float(threshold)
    masks[f"over_{previous:g}px"] = valid & (d0_error > previous)
    return masks


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


def _batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _reduce_stats(stats: Mapping[str, float], device: torch.device) -> Dict[str, float]:
    keys = sorted(stats)
    values = torch.tensor([stats[key] for key in keys], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return dict(zip(keys, values.cpu().tolist()))


def _metric(abs_sum: float, bad3_count: float, count: float) -> Dict[str, float]:
    if count <= 0:
        return {"epe_pixel_mean": float("nan"), "out3_pixel_percent": float("nan")}
    return {
        "epe_pixel_mean": abs_sum / count,
        "out3_pixel_percent": 100.0 * bad3_count / count,
    }


def _model_args(config: Mapping[str, object], corr_radius: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_downsample=int(config.get("n_downsample", 2)),
        n_gru_layers=int(config.get("n_gru_layers", 3)),
        hidden_dims=list(config.get("hidden_dims", [128, 128, 128])),
        context_norm=str(config.get("context_norm", "instance")),
        corr_radius=int(corr_radius),
    )


def _validate_checkpoint(checkpoint, state, args) -> Mapping[str, object]:
    config = checkpoint.get("model_config", {})
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint is missing model_config metadata")
    if config.get("model") != "defom_pivno":
        raise ValueError(
            f"expected a defom_pivno checkpoint, got model={config.get('model')!r}"
        )
    if int(config.get("pivno_input_channels", 0)) != 3:
        raise ValueError("this evaluator expects the RGB defom_pivno checkpoint")
    fuse_weight = state.get("refine_right_fuse.0.weight")
    if fuse_weight is None or fuse_weight.ndim != 4:
        raise KeyError("checkpoint is missing refine_right_fuse.0.weight")
    expected_channels = 32 * (1 + 3 * (2 * int(args.corr_radius) + 1))
    if int(fuse_weight.shape[1]) != expected_channels:
        raise ValueError(
            "corr_radius/checkpoint mismatch: "
            f"expected fuser input {expected_channels}, got {fuse_weight.shape[1]}"
        )
    return config


def _initialize_stats(stage_names: Sequence[str], bin_names: Sequence[str]):
    stats = defaultdict(float)
    stats["meta/images"] = 0.0
    stats["meta/valid_pixels"] = 0.0
    for stage in stage_names:
        stats[f"global/{stage}/abs_sum"] = 0.0
        stats[f"global/{stage}/bad3"] = 0.0
        stats[f"global/{stage}/image_epe_sum"] = 0.0
    for bin_name in bin_names:
        stats[f"bin/{bin_name}/count"] = 0.0
        stats[f"bin/{bin_name}/improved"] = 0.0
        stats[f"bin/{bin_name}/worsened"] = 0.0
        stats[f"bin/{bin_name}/worsened_0p5"] = 0.0
        for stage in stage_names:
            stats[f"bin/{bin_name}/{stage}/abs_sum"] = 0.0
            stats[f"bin/{bin_name}/{stage}/bad3"] = 0.0
    for region in ("edge", "non_edge"):
        stats[f"region/{region}/count"] = 0.0
        for stage in ("d0", stage_names[-1]):
            stats[f"region/{region}/{stage}/abs_sum"] = 0.0
            stats[f"region/{region}/{stage}/bad3"] = 0.0
    return stats


@torch.no_grad()
def evaluate(args) -> Dict[str, object]:
    distributed, rank, world_size, device = _distributed_info(args.device)
    is_main = rank == 0
    try:
        checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
        state = _checkpoint_state(checkpoint)
        preliminary_config = checkpoint.get("model_config", {})
        model_args = _model_args(preliminary_config, args.corr_radius)
        config = _validate_checkpoint(checkpoint, state, model_args)
        model = DEFOMStereo(model_args)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        max_disp = float(args.max_disp if args.max_disp is not None else config["max_disp"])
        requested_iterations = sorted({
            int(value) for value in args.report_iters if 0 < int(value) <= args.iters
        })
        if args.iters not in requested_iterations:
            requested_iterations.append(args.iters)
        stage_names = ["d0", *[f"iter_{value}" for value in requested_iterations]]
        final_stage = stage_names[-1]
        bin_names = [
            *[f"{lower:g}_{upper:g}px" for lower, upper in zip(
                (0.0, *D0_ERROR_THRESHOLDS[:-1]), D0_ERROR_THRESHOLDS
            )],
            f"over_{D0_ERROR_THRESHOLDS[-1]:g}px",
        ]
        stats = _initialize_stats(stage_names, bin_names)

        dataset = SceneFlowDatasets(
            root=args.sceneflow_root,
            dstype="frames_finalpass",
            things_test=True,
        )
        indices = list(range(rank, len(dataset), world_size))
        if is_main:
            print(
                f"checkpoint={args.restore_ckpt} step={checkpoint.get('step', 'unknown')} "
                f"images={len(dataset)} iters={args.iters} stages={stage_names} "
                f"world_size={world_size} batch_per_gpu={args.batch_size}"
            )

        progress = tqdm(
            list(_batched(indices, args.batch_size)),
            disable=not is_main,
            desc="PIVNO refinement",
        )
        for batch_ids in progress:
            samples = [dataset[index] for index in batch_ids]
            shapes = {tuple(sample["img1"].shape) for sample in samples}
            if len(shapes) != 1:
                raise ValueError(f"unequal image shapes in batch {batch_ids}: {sorted(shapes)}")
            image1 = torch.stack([sample["img1"] for sample in samples]).to(device)
            image2 = torch.stack([sample["img2"] for sample in samples]).to(device)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            with torch.cuda.amp.autocast(
                enabled=args.mixed_precision and device.type == "cuda"
            ):
                init_predictions, recurrent_predictions = model(
                    image1,
                    image2,
                    iters=args.iters,
                    test_mode=False,
                )

            predictions = {"d0": padder.unpad(init_predictions[-1]).float().cpu()}
            for iteration in requested_iterations:
                predictions[f"iter_{iteration}"] = padder.unpad(
                    recurrent_predictions[iteration - 1]
                ).float().cpu()

            for batch_index, sample in enumerate(samples):
                ground_truth = sample["disp"].float()
                finite_gt = torch.isfinite(ground_truth)
                safe_gt = torch.where(finite_gt, ground_truth, torch.zeros_like(ground_truth))
                valid = (
                    (sample["valid"] >= 0.5)
                    & finite_gt
                    & (safe_gt >= 0.0)
                    & (safe_gt < max_disp)
                )
                if not bool(valid.any()):
                    continue

                errors = {}
                for stage, prediction_batch in predictions.items():
                    prediction = prediction_batch[batch_index]
                    if prediction.shape != ground_truth.shape:
                        raise ValueError(
                            f"{stage} shape mismatch for {sample['imageL_file']}: "
                            f"{tuple(prediction.shape)} vs {tuple(ground_truth.shape)}"
                        )
                    if not bool(torch.isfinite(prediction[valid]).all()):
                        raise FloatingPointError(
                            f"non-finite {stage} prediction for {sample['imageL_file']}"
                        )
                    errors[stage] = (prediction - safe_gt).abs()

                stats["meta/images"] += 1
                stats["meta/valid_pixels"] += int(valid.sum())
                d0_bins = build_d0_error_masks(errors["d0"], valid)

                for stage, error in errors.items():
                    valid_error = error[valid]
                    stats[f"global/{stage}/abs_sum"] += float(valid_error.sum())
                    stats[f"global/{stage}/bad3"] += int((valid_error > 3.0).sum())
                    stats[f"global/{stage}/image_epe_sum"] += float(valid_error.mean())
                    for bin_name, bin_mask in d0_bins.items():
                        bin_error = error[bin_mask]
                        stats[f"bin/{bin_name}/{stage}/abs_sum"] += float(bin_error.sum())
                        stats[f"bin/{bin_name}/{stage}/bad3"] += int((bin_error > 3.0).sum())

                final_error = errors[final_stage]
                for bin_name, bin_mask in d0_bins.items():
                    count = int(bin_mask.sum())
                    stats[f"bin/{bin_name}/count"] += count
                    stats[f"bin/{bin_name}/improved"] += int(
                        (final_error[bin_mask] < errors["d0"][bin_mask]).sum()
                    )
                    stats[f"bin/{bin_name}/worsened"] += int(
                        (final_error[bin_mask] > errors["d0"][bin_mask]).sum()
                    )
                    stats[f"bin/{bin_name}/worsened_0p5"] += int(
                        (final_error[bin_mask] > errors["d0"][bin_mask] + 0.5).sum()
                    )

                edge, _, _ = disparity_edge_mask(
                    safe_gt.unsqueeze(0),
                    valid.unsqueeze(0),
                    edge_mode="topk",
                    edge_topk=args.edge_topk,
                    edge_dilation=args.edge_dilation,
                )
                edge = edge[0].bool() & valid
                region_masks = {"edge": edge, "non_edge": valid & ~edge}
                for region, region_mask in region_masks.items():
                    count = int(region_mask.sum())
                    stats[f"region/{region}/count"] += count
                    for stage in ("d0", final_stage):
                        region_error = errors[stage][region_mask]
                        stats[f"region/{region}/{stage}/abs_sum"] += float(region_error.sum())
                        stats[f"region/{region}/{stage}/bad3"] += int(
                            (region_error > 3.0).sum()
                        )

            del init_predictions, recurrent_predictions, predictions

        stats = _reduce_stats(stats, device)
        image_count = stats["meta/images"]
        valid_count = stats["meta/valid_pixels"]
        if image_count <= 0 or valid_count <= 0:
            raise RuntimeError("evaluation produced no valid images or pixels")

        global_results = {}
        for stage in stage_names:
            stage_metric = _metric(
                stats[f"global/{stage}/abs_sum"],
                stats[f"global/{stage}/bad3"],
                valid_count,
            )
            stage_metric["epe_image_mean"] = (
                stats[f"global/{stage}/image_epe_sum"] / image_count
            )
            global_results[stage] = stage_metric

        bin_results = {}
        for bin_name in bin_names:
            count = stats[f"bin/{bin_name}/count"]
            stage_results = {}
            for stage in stage_names:
                stage_results[stage] = _metric(
                    stats[f"bin/{bin_name}/{stage}/abs_sum"],
                    stats[f"bin/{bin_name}/{stage}/bad3"],
                    count,
                )
            bin_results[bin_name] = {
                "pixels": int(count),
                "pixel_percent": 100.0 * count / valid_count,
                "stages": stage_results,
                "final_improved_pixel_percent": 100.0 * stats[f"bin/{bin_name}/improved"] / max(count, 1.0),
                "final_worsened_pixel_percent": 100.0 * stats[f"bin/{bin_name}/worsened"] / max(count, 1.0),
                "final_worsened_over_0p5px_percent": 100.0 * stats[f"bin/{bin_name}/worsened_0p5"] / max(count, 1.0),
            }

        region_results = {}
        for region in ("edge", "non_edge"):
            count = stats[f"region/{region}/count"]
            region_results[region] = {
                "pixels": int(count),
                "pixel_percent": 100.0 * count / valid_count,
                "d0": _metric(
                    stats[f"region/{region}/d0/abs_sum"],
                    stats[f"region/{region}/d0/bad3"],
                    count,
                ),
                "final": _metric(
                    stats[f"region/{region}/{final_stage}/abs_sum"],
                    stats[f"region/{region}/{final_stage}/bad3"],
                    count,
                ),
            }

        results = {
            "checkpoint": str(Path(args.restore_ckpt).resolve()),
            "checkpoint_step": checkpoint.get("step"),
            "dataset": "FlyingThings3D TEST frames_finalpass",
            "max_disp": max_disp,
            "images": int(image_count),
            "valid_pixels": int(valid_count),
            "iterations": args.iters,
            "reported_stages": stage_names,
            "d0_bin_definition": "mutually exclusive full-resolution absolute d0 error",
            "edge_definition": {
                "mode": "topk_gt_disparity_gradient",
                "topk": args.edge_topk,
                "dilation": args.edge_dilation,
            },
            "global": global_results,
            "d0_error_bins": bin_results,
            "regions": region_results,
        }
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
    parser.add_argument("--max_disp", type=float, default=None)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--report_iters", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--edge_topk", type=float, default=0.10)
    parser.add_argument("--edge_dilation", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1, help="batch per GPU")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output_json")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if not 0.0 <= args.edge_topk <= 1.0:
        parser.error("--edge_topk must be in [0, 1]")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(parse_args())
