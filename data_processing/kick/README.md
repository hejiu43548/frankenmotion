# Kick source repair

Current candidate: Betail
`/mnt/sda2/frankenmotion/outputs_amass/source_repair_kick_20261009_v1`.
It contains33train/2val events across28/2 original recordings; user review is
pending. Other19 task records are unchanged, including provisionally accepted
wave and strike. No training has started.

Run the Hydra-configured tools from the repository root:

```sh
python -m data_processing.kick.repair
python -m unittest data_processing.kick.test_rules
python -m data_processing.kick.review
python -m data_processing.parameter_coverage \
  dataset_directory=/mnt/sda2/frankenmotion/outputs_amass/source_repair_kick_20261009_v1 \
  output=experiments/kick_source_repair_20261009/coverage
```

The configurations are `config/kick_repair.yaml`, `config/kick_review.yaml`, and
`config/parameter_coverage.yaml`. Large caches/videos remain in `outputs_amass`;
small reports, configuration snapshots and coverage records go in
`experiments/kick_source_repair_20261009/`.

## Source and crop policy

Require BABEL's explicit category `kick`. Prefer frame-level events and HDM05
scene03-02 recordings. Single-action sequence annotations are also eligible only
when `mul_act=false` and every sequence label passes the kick category/style
checks. They are explicitly recorded as `single_action_sequence`, not mislabeled
as frame-level annotations. No caption-only regex admission is used. The native
HDM05 cut mapping remains unavailable; HDM05 sources here use BABEL boundaries.

Preserve existing Frankenstein family splits and reject test/unassigned families.
Reject explicit left, side, backwards, round, spinning or mixed-leg labels.
Locate right-ankle forward-reach peaks in the body-facing coordinates, inside the
source event. Try real60/50/40-frame crops and peak positions30/24/18. Require
actual forward excursion, ankle lift, limited lateral deviation, bounded root
and supporting-foot movement, little heading change, and no left-kick peak in
the crop. Reject conflicting context annotations and overlapping same-source
events. No retiming, mirroring, stabilization, padding or amplitude scaling.

Recompute the unchanged legacy quantity: maximum canonical forward displacement
of the right ankle relative to pelvis in frames10:51, minus its initial value,
with the existing human-height scale. The initial pose affects this metric;
it must not be interpreted as an independent kick angle or absolute toe reach.
Original source-event text is encoded with the existing CLIP/PCA; only action
and right-leg intervals receive local text. Unspecified parts remain masked.

## Parameter-supervision limitation

Requested range:0.25–0.70m. Actual training quantities span0.434–1.044m; only10
of33 lie inside the requested range. Five of ten equal-width requested-range
bins are empty. Validation contains2 samples at0.721–0.801m, both outside the
requested range. Source semantics and support of the command range are separate
requirements: this candidate does **not** establish adequate parameter supervision.
Do not fill the missing bins by changing labels or quietly scaling motions.

Review sampling splits training records into ten equal-count amplitude groups
and draws one record from each with seed20261009, preferring different families.
The videos show unmodified training-cache FK at20fps. Inspect purity, preparation,
recovery, supporting foot and the ordinary forward-kick style; automated rules
are not a substitute for manual semantic approval.

The35 caches are checked against their exact source slices and recomputed
quantities. Save policy/configuration, source and cache hashes, failed crop
attempts, source annotation IDs, code snapshot and review metadata with each run.
