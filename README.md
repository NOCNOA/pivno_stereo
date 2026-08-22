# PIVNO stereo bundle

This directory collects the PIVNO stereo source code, selected trained model
weights, and the corresponding training/evaluation logs from
`defom_pact_portable`. Repository-relative source paths are preserved so the
source files can be copied into a matching checkout.

## Source code

- `PIVNO/`: upstream PIVNO source, configs, datasets, and list files.
- `core/pivno_models/`: PACT-PIVNO, DEFOM-PIVNO, and gated DEFOM-PIVNO.
- Shared modules used directly by the integrated models.
- `train_stereo.py` and `evaluate_stereo.py` integration entrypoints.
- PIVNO loss, training/evaluation scripts, diagnostic tools, and tests.

## Model weights

The files under `weights/` are model-only checkpoints and do not contain the
optimizer state.

| Directory | Model | Checkpoint role |
| --- | --- | --- |
| `weights/pact_pivno/` | PACT-PIVNO | completed 200k model |
| `weights/defom_pivno/` | DEFOM-PIVNO | completed 200k model |
| `weights/defom_pivno_gated_ft20k/` | gated DEFOM-PIVNO | completed 20k warm-start fine-tune |
| `weights/defom_pivno_gated_full200k/` | gated DEFOM-PIVNO | completed 200k model plus the 190k best-EPE checkpoint |

FlyingThings full-validation reference metrics (`0 <= disp < 768`, 32
refinement iterations):

| Model | EPE | Out3.0 |
| --- | ---: | ---: |
| PACT-PIVNO 200k | 1.193270 | 3.870974% |
| DEFOM-PIVNO 200k | 1.149137 | 3.796849% |
| gated DEFOM-PIVNO warm-start 20k | 1.131212 | 3.762349% |
| gated DEFOM-PIVNO full 200k | 1.190846 | 3.786459% |
| gated DEFOM-PIVNO full 190k | 1.158070 | 3.802128% |

## Logs

- `logs/pact_pivno/`: initial run, resumed 200k training, and full evaluation.
- `logs/defom_pivno/`: completed 200k training.
- `logs/defom_pivno_gated_ft20k/`: completed warm-start fine-tuning.
- `logs/defom_pivno_gated_full200k/`: completed from-start 200k training.

## Not included

- Optimizer-bearing `checkpoint_latest.pth` files and all periodic weights.
- SceneFlow and other datasets.
- TensorBoard outputs, generated evaluation artifacts, Python caches, and
  nested Git repositories.

This is a movable source-and-artifact directory, not a standalone Python
environment. Dependencies and datasets must be provided in the target
environment.
