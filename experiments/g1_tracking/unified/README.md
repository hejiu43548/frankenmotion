# One shared FrankenMotion → G1 tracker

This experiment replaces the previous per-task SONIC/BeyondMimic controller bank
with one shared BeyondMimic-style actor. Every motion uses the same observation
construction, normalization, MLP, action scale, and native robot dynamics. There
is no task ID, controller selection, checkpoint selection, or mixture of experts
inside the tracker. Only one globally selected development checkpoint is eligible
for final evaluation.

The scope is **tracker unification**. The source generator remains the frozen V3
adapter selection, and GMR retargeting retains the frozen raise-hand/lean geometric
refinements. This does not claim a completely task-independent upstream pipeline,
new arbitrary-text generalization, or hardware deployment.

## Training changes

The shared actor has 29 action outputs and hidden layers 512/256/128 (ELU). The
original 160-dimensional actor observations can be extended with future reference
joints, joint velocities, and relative anchor position/orientation. Short preview
uses 0.1, 0.2 and 0.4 seconds (361 total inputs); long preview also includes 0.7
and 1.0 seconds (495 total inputs). Future samples clamp to the current clip end.
The actor does not receive a motion label or a prompt embedding.

Finite references now end through the termination manager, so PPO receives an
actual terminal transition. The former multi-motion command reset could silently
teleport between references without a terminal flag. All-environment boundary
smokes and finite/index checks are required before full training.

Training uses 735 native 50 Hz clips: 495 individual motions, 160 concatenated
sequences and 80 locomotion/arm compositions. Thirteen metadata groups receive
balanced environment slots; labels are used only by the data sampler. Each reset
starts at the clip beginning with probability 0.35, otherwise at a random phase.
Thirty-second episodes accommodate long sequence references. Internal sequence
boundaries do not reset simulation or load another policy.

The improved common loss includes body-position tracking, narrow global anchor
position tracking and a joint reward combining mean accuracy across all 29 joints
with mean accuracy across the eight worst-tracked joints. This is the same formula
for every motion. Root-orientation/wide-position reward variants are also tested.
See each run's `protocol.json` for exact weights rather than assuming all ablations
use the improved loss.

The pretrained observation normalizer has roughly 4.9 billion accumulated samples.
New preview features otherwise barely update during a short fine-tune. Preview
normalization is therefore calibrated on training reset states only. A compensating
first-layer affine transformation preserves the loaded policy function before
learning. Existing proprioceptive normalization is retained. Pointwise and random
input checks test the compensation; this is not itself evidence of better tracking.

## Evaluation boundaries

Development selection uses 110 requests: eleven tasks, five numeric commands and
two source noise seeds. The final cohort reserves 880 requests: the same eleven
tasks/five commands, four cached prompt templates and four new source noise seeds.
Training, development and final seed families are disjoint. Templates and command
ranges overlap, so the final test measures fresh-noise generalization within this
protocol, not unseen-language or out-of-range command extrapolation.

A model is selected by lowest macro semantic `E_all` on all eleven development
tasks. For a physically complete, semantically valid motion, the error is absolute
command error divided by that task's command span and capped at one. Physical or
semantic failure is assigned one. Every task has equal macro weight. Joint pass
requires both the task's event check and its established numeric tolerance. A robot
remaining upright is not sufficient: for example, standing through a requested
jump can be physically complete but must fail the jump event check.

Reference and executed quantities are independently recomputed from saved raw
states using the established metric implementation. Human-equivalent distance and
speed units account for the morphology scale; angles stay in radians. The shared
policy and frozen routed V3 baseline receive exactly the same source/reference
requests. Their native control/actuator configurations differ, so that comparison
is a whole-pipeline comparison. A separate native BeyondMimic routed control helps
expose this confound and is never eligible as the unified final policy.

Continuous-sequence/composition tests report full-reference completion and all
component event/precision checks. A failed later component cannot be hidden by
reporting an earlier successful segment. One checkpoint is loaded for the entire
rollout, with no internal-boundary reset.

## Server-oriented reproduction

The configured root is `/home/pku/frankenmotion`. This branch builds on the prior
`codex/g1-parameterized-tracking` experiment code and requires its external model,
reference and simulator assets. It is not a self-contained installer. Large
weights, input caches and trajectories stay on the server; provenance manifests
and hashes accompany the final selected artifact.

Runtime environments:

- `.conda/bin/python`: source diffusion generation.
- `work/mjlab_stable_env/bin/python`: native GPU tracker training/evaluation and actor export.
- `work/g1_sim_env/bin/python`: CPU retargeting and raw-state metric audits.

New entrypoint: `work/unified_run_command_20261004.py`. It uses the frozen shared
tracker by default and supports separate generate/retarget/simulate stages. Each
request needs a new name; existing artifacts are not overwritten. Development
checkpoint overrides are explicit and reject the reserved final source seeds.
Task/command arguments describe the requested source motion; they do not select
the tracker. The entrypoint removes inherited tracker-routing environment settings.

```bash
cd /home/pku/frankenmotion
.conda/bin/python work/unified_run_command_20261004.py --task wave --command 0.15 --seed 95041000 --name unified_wave_example
.conda/bin/python work/unified_run_command_20261004.py --task walk --command 0.8 --seed 95041000 --name unified_walk_example
```

These default commands require the final `frozen_unified/protocol.json` and its
hash-verified `policy.pt`. The actor-only TorchScript export contains normalization
and one MLP, but still requires the trained observation order, native action scaling,
joint order and robot configuration. It is not a validated hardware controller.

The paired continuation driver, `work/unified_run_pair_20261004.py`, records its
parent selection and all arguments before launching the two runs. Both use the
same globally selected V4 parent, common rewards/corpus, fresh optimizer and seed;
only the future preview horizon differs. Training uses one seed per variant, so
this is exploratory evidence rather than a multi-training-seed significance claim.
