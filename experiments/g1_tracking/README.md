# FrankenMotion → G1 parameterized tracking experiments

This branch preserves the experiment code and frozen routing protocols from the
2026-10-03/04 study. It is a server-oriented research pipeline, not a standalone
installer or a hardware controller. Existing baseline and experimental outputs
remain on the server. Large assets and generated trajectories are not committed.

## Final V3 pipeline

Numeric task commands enter diffusion sampling directly; human outputs are not
edited after generation. The original root adapter conditions speed/yaw rate.
The task and physical/time adapters extend conditioning to eleven measured tasks.

| Stage | Frozen selection |
| --- | --- |
| Generation | Original task adapter for kick; single-jump supervised adapter for jump; physical/time adapter otherwise |
| Retarget | GMR uniform; source-derived wrist/pelvis constraints for raise_hand and torso orientation constraints for lean |
| Tracker | BeyondMimic for reach, sidestep, back_walk; SONIC v1.1 mode 0 otherwise |
| BM weights | Dance checkpoint for reach; development-trained side_std01 and back_multi for lateral/backward motion |

This is a task-specific checkpoint bank. Seamless online switching is untested.
The later turn and strike specialist training attempts failed the preregistered
development gate and are **not** deployed in V3.

## Run on the configured server

```bash
cd /home/pku/frankenmotion
.conda/bin/python work/run_command_v3_20261003.py --task kick --command 0.5875 --seed 91023004 --name new_kick_request
.conda/bin/python work/run_command_v3_20261003.py --task sidestep --command 0.8 --seed 91023004 --name new_side_request
```

Use a fresh `--name`; existing requests are preserved. `--prompt-index 0..3`
selects a cached template, not arbitrary free text. Results are written below
`outputs_amass/franken_improve_20261003/single_requests/<name>/`.
`result.json` measures actual simulation; generation.json records the source
checkpoint and prompt. Generation uses `.conda/bin/python`; SONIC simulation
uses `work/g1_sim_env/bin/python`; BM uses `work/mjlab_stable_env/bin/python`.

Commands use human-equivalent distances/speeds, not literal G1 distances.

| Task | Command range | Unit |
| --- | --- | --- |
| raise_hand | 0.35–0.75 | m |
| reach | 0.25–0.55 | m |
| strike | 1.5–2.5 | m/s |
| wave | 0.08–0.22 | m |
| turn | 0.45–1.5 | rad |
| sidestep | 0.4–1.2 | m |
| back_walk | 0.35–0.9 | m/s |
| kick | 0.25–0.7 | m |
| jump | 0.25–0.55 | m |
| lean | 0.4–0.9 | rad |
| walk | 0.5–1.1 | m/s (XY path speed; direction checked separately) |

## Source map

- `outputs_amass/root_control_20260925/code/`: original root-control adapter.
- `outputs_amass/franken_eleven_20261003/code/`: task adapter, generation,
  source representation, metrics, and original retarget/tracking baseline.
- `work/physical_adapter_20261003.py`, `jump_aligned_adapter_20261003.py`:
  generation training extensions.
- `work/gmr_probe_20261003.py`, `task_retarget_v2_20261003.py`: retargeting.
- `work/mjlab_finetune_20261003.py`, `mjlab_specialist_20261003.py`: BM training.
- `work/run_command_v3_20261003.py`: supported single-request V3 entrypoint.
- `work/*capacity_20261003.py`: corrected BM execution/assessment scripts.
- `outputs_amass/franken_improve_20261003/frozen_*`: frozen protocols;
  weight hashes are committed, weight binaries are not.

Python sources under the normally ignored output directories are intentionally
tracked to preserve their original import paths. New generated outputs stay ignored.

## Reproduction dependencies

Paths currently assume `/home/pku/frankenmotion`. A checkout on another machine
needs consistent path configuration and the existing assets: base model, root/task
adapters, frozen controller checkpoints, skeleton, cached text embeddings, AMASS
training data, SONIC ONNX models and G1 robot meshes. They remain in the configured
server's `outputs_amass/` and `work/` directories. Merely cloning this branch does
not install these assets. The fuller reproduction archive on the server is
`outputs_amass/franken_improve_20261003/frankenmotion_g1_reproduction_v3.tar.gz`;
it includes frozen task/controller weights and provenance, but still excludes
large base models, datasets, third-party repositories and raw rollouts.

Pinned external components:
- GMR: `bb1bbe40774794fceb2a7c579a3464a28e68c844`
- mjlab v1.1.1: `3cb20cb64507f4d1cf5c8f271ff79b5144727025`
- SONIC v1.1, mode 0; MuJoCo-Warp 3.5.0, Torch 2.7.0, Warp 1.12.0

The frozen protocol hashes must remain consistent with the exact assets used.
Historical batch drivers have fixed output paths; do not rerun them over archived
benchmarks. `finish_*` scripts are historical job watchers with original process
IDs, not reusable launchers. Reproduce batch experiments in an isolated output
copy with deliberate path changes. Use the single-request entrypoint for normal
server testing.

## Evidence and limitations

The corrected first-round study contains 880 paired requests and 3520 physical
rollouts. It compares old generation + direction IK, old generation + GMR,
new generation + GMR + SONIC, and new generation + GMR + BM.

The independent integrated V3 cohort uses 11 tasks × 4 shared text templates ×
4 fresh noises × 5 commands = 880 requests. Joint numeric-and-event passes improve
from 277/880 to 436/880 against the previous selected pipeline on the same inputs.
Eight unchanged tasks share execution records; three changed tasks have separate
paired controls, totaling 1120 unique rollouts. Do not mix these counts with the
first-round cohort. P10–P90 bands describe source variation, not confidence bounds.

A BM contact-buffer overflow was discovered in historical evaluations. Corrected
runs use nconmax=256, njmax=2048 with unchanged inputs, weights and thresholds.
Use `_capacity256` result folders; original results are historical. The corrected
3520- and 1120-rollout raw-state audits have zero metric recomputation discrepancy
and no overflow warnings. Different native actuator/simulator configurations mean
SONIC/BM comparisons concern whole pipelines, not only policy networks.

Strike speed, turn bias and high jumps remain imperfect. A separate paraphrase
stress test achieved only 23/110 joint passes; shared-template results do not imply
arbitrary-text generalization. No hardware commands were sent.

Research: [GMR](https://arxiv.org/html/2510.02252v1),
[BeyondMimic](https://arxiv.org/html/2508.08241v2),
[OmniRetarget](https://arxiv.org/html/2509.26633v3).
OmniRetarget informed the contact-constraint investigation; it was not reproduced.
