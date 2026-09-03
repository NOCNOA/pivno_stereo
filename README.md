# PIVNO stereo portable bundle

This directory was refreshed from `/home/yijiayi/defom_pact_portable` on
2026-09-03. It keeps repository-relative paths so the code can be run here or
copied into a compatible checkout.

## Included

- `PIVNO/`: upstream PIVNO implementation, configs, dataset loaders, and list
  files. The nested Git repository, legacy `save/` checkpoint, and caches are
  excluded.
- `core/pivno_models/`: all integrated PIVNO model variants currently present
  in the source checkout.
- Shared `core/`, `utils/`, and `depth_anything_v2/` source required by the
  current training/evaluation entrypoints.
- `train_stereo.py`, `evaluate_stereo.py`, PIVNO launch scripts, diagnostic
  tools, and focused tests.
- `weights/current_exports/`: model-only exported checkpoints from the source
  checkout. Optimizer-bearing `checkpoint_latest.pth` and periodic checkpoints
  are deliberately excluded.
- `logs/current_exports/`: matching PIVNO training/evaluation logs.
- `evaluation_results/`: compact PIVNO JSON summaries.
- The existing isolated `defom_pivno_mobilenetv2` implementation and its
  train/eval registration are retained.

## Integrated PIVNO model names

- `pact_pivno`
- `defom_pivno`
- `defom_pivno_mobilenetv2`
- `defom_pivno_gated`
- `defom_pivno_gated_gru1`
- `defom_pivno_gated_gru3`
- `defom_pivno_gated_gru_kernel_ablation`
- `defom_pivno_gwc4_enc16_concat_gru3`
- `defom_pivno_gwc4_enc16_concat_gru3_mask_sr`
- `defom_pivno_gated_gru3_gwc4_mask_sr`
- `defom_pivno_gated_gru3_gwc4_mask_rgb_sr`
- `defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr`
- `defom_pivno_gated_gru3_gwc4_mask_last_delta_sr`
- `defom_pivno_gated_gru3_gwc4_last_delta_direct_sr`

Use each checkpoint only with the matching `--model` value and architecture
metadata. A successful strict state-dict load does not make checkpoints from
different variants interchangeable.

## Not included

- SceneFlow or other image/disparity datasets.
- Optimizer/scheduler training-state checkpoints and periodic weights.
- TensorBoard runs, generated visualizations, Python caches, or nested Git
  metadata.
- Pretrained third-party weights required by non-PIVNO branches.

`MANIFEST.txt` lists the packaged files, `SHA256SUMS` records checksums for the
selected exported model weights, and `TEST_REPORT.md` records the verification
boundary and the two pre-existing test mismatches.
