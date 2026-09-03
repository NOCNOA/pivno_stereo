# Package verification report

Verification date: 2026-09-03

- Python byte-compilation: passed for packaged source, entrypoints, tools, and
  tests.
- Import smoke test: `train_stereo`, `evaluate_stereo`, and the isolated
  MobileNetV2 PIVNO model imported successfully in the `defomstereo` conda
  environment.
- Packaged `unittest` suite: 87 tests run; 85 passed, 1 failed, and 1 errored.
- All packaged SR variants, including last-delta direct SR, last-delta weighted
  SR, RGB-hidden SR, and MobileNetV2 tests passed.

The two failures are pre-existing source/test drift in
`tests/test_defom_pivno_gated_gru3.py`: those tests assume
`MATCH_NUM_GROUPS == 4` and construct four group-correlation tensors, while the
current model default resolves to 8 groups. Re-running only those two tests in
the source checkout produced the same failure and error. The package did not
introduce this mismatch.

No GPU forward pass or full-dataset evaluation was performed as part of this
packaging operation.
