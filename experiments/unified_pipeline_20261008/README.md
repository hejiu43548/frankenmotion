# Unified FrankenMotion commands and shared G1 tracker (2026-10-08)

This source snapshot supersedes the older gait/table demo on this branch. It includes unified command conditioning, training/export, reference and physical evaluation, the current table demo, and an unsuccessful temporal-overlap experiment. Historical scripts are retained for provenance, not selected automatically at inference.

## Selected pipeline

Text/time assignments + numeric commands → frozen FrankenMotion backbone with a learned shared command controller → human motion → GMR/retarget and reference preparation → one shared G1 tracker → MuJoCo dynamics.

- `work/unified_commands_20261007/single_runtime.py` loads one complete generator checkpoint. `unified_control.py` implements shared conditioning; `command_schema.py` and `command_inputs.py` define numeric fields. The 46-dimensional condition includes task/intent, task value, speed/yaw rate, endpoint distance/direction, reach height, validity flags and time features. Shared residuals enter four denoiser layers and the output. The backbone remains frozen during controller training.
- `train_features.py`, `legacy_targets.py`, `export_full.py` implement training and export. Older root/task/goal/reach/exit adapters are teacher or baseline dependencies; they are not additional runtime checkpoints in the standalone exported generator.
- The selected tracker is one TorchScript actor containing its frozen base, learned reference-conditioned residual and normalization. It does not select weights by action category. Generator and tracker are still two separate models.
- `work/universal_tracker_20261006/` and `work/jump_tracker_20261007/` retain tracker training, conversion, export and evaluation. `work/unified_generator_20261007/` records the earlier task-unification stage.

## Runtime assets and environment

See `assets.json` for the exact selected server paths and SHA256 hashes. Checkpoints, datasets, SMPL/SMPL-H licensed assets, text encoder assets, prompt caches, videos, simulator models and third-party installations are external. Code under `outputs_amass/*/code` is intentionally included because historical scripts import it.

This is a configured-server research snapshot, not a portable fresh-clone installation. Most experiment drivers assume `/home/pku/frankenmotion`. Generation uses `.conda/bin/python`; simulation uses `work/mjlab_stable_env/bin/python`. Existing GMR, mjlab, MuJoCo and robot assets must be restored separately. Historical training also requires its recorded teacher assets and training data. Do not launch all queue scripts: they represent different experimental stages.

## Inference entries

Standalone motion inference (prompt cache contains `local` and `tx` text-conditioning tensors):

```bash
cd /home/pku/frankenmotion
.conda/bin/python work/unified_commands_20261007/infer.py --help
```

Current table pipeline, using a fresh output name:

```bash
cd /home/pku/frankenmotion
.conda/bin/python work/unified_commands_20261007/scene_queue.py \
  --name table_reproduction_001 --full --layouts 2 --seed 10808001 \
  --checkpoint outputs_amass/unified_commands_20261007/exports/candidate_v4/unified_generator.pt
```

The queue explicitly passes the shared tracker from `backup/tracker.pt`. It generates walking, reaching/retraction, turning and departure segments. It also applies bounded command calibration, GMR, walking retiming, rigid placement, blends and reference anchoring at phase transitions. Thus the table demo is not an unprocessed single diffusion sample. The actual robot trajectory comes from policy actions and simulation, not reference playback or repeated robot-state assignment.

## Evidence and limitations

- The merged-control 880-case assessment produced 669 human-reference, 707 G1-reference and 561 physical-execution passes under the recorded metrics. Its paired baseline produced 675/708/563. Scalar metric passes do not establish naturalness or complete action semantics. Jump tracking and other task weaknesses remain.
- Four current table scenes (two layouts, each with 0.3 m and 0.5 m reach commands) completed the recorded checks. These are demonstrations, not broad hardware or robustness validation.
- `work/merged_report_*_20261008.py` contains re-evaluation and structure checks. `structure_audit.json` records the independently loaded unified checkpoint audit.
- `work/temporal_mix_*_20261008.py` tests walking at 1–5 s, left waving at 2–4 s and right waving at 3–5 s. This experiment did NOT accurately achieve the requested timed composition: reference timing/overlap was imperfect and physical tracking reduced arm motion. Complete simulation rollout is not command success.
- No new tracker adjustment was completed after this selected checkpoint during the latest demo review. Older SONIC/BeyondMimic-related teacher/baseline scripts are historical dependencies, not an inference-time weight switch.

## Snapshot validation

All 652 copied Python source files parsed successfully; Trailing whitespace was normalized in the publication copy; source hashes are in `source_manifest.json`. This publication does not rerun training or the full simulation benchmark. Existing audits are retained rather than represented as fresh tests. Large assets remain on the server.
