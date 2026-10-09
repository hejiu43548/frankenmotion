# HDM05 sidestep source repair

Current candidate: `source_repair_sidestep_20261010_v2`, 34 train /6 val clips
from10/2 original recordings. Only sidestep is active; all other tasks are deferred.
No sidestep training has been launched. Ten source-paired TRAIN videos await review.

## Source boundary status

The [official01-01 script](https://resources.mpi-inf.mpg.de/HDM05/01-01/index.html)
defines phase6 as sideways left without crossing the feet. The [class table](https://resources.mpi-inf.mpg.de/HDM05/cuts/index.html)
lists walkLeft2Steps and walkLeft3Steps, separately from circular walking and
rightward crossover steps. The official mocapTakes2Cuts mapping returned HTTP403
on2026-10-10 locally, on Betail and through the browsing service. **These are
script/motion supplemented boundaries, not verified official2/3-step cut rows.**

`inspect_sources.py` inventories only scene01-01; existing source-family splits
are retained. A test family is listed but never inspected or admitted. Each
available train/val recording must have exactly one unambiguous body-local left
run, passing direction and noncrossing checks. BABEL frame labels with act_cat
are preserved as corroborating evidence, not substituted for native cut IDs.
No non-HDM05 source or generic text-regex candidate is admitted.

`repair.py` uses ankle-distance minima to delimit complete opening/closing cycles.
Clips are22–31 frames at20fps. Neighboring cycles share only their closure frame.
The initial cycle of bk01-01 take03 is rejected for turning/fore-aft motion.
The rest remain pending human semantic review.

## Direction and mirroring

Existing task5 measures `-min(root_y - initial_root_y)`: positive anatomical-right
travel in the canonical frame. Feeding native left motion with a positive
left-distance label would silently corrupt this supervision.

`rules.py` swaps22 SMPL body joints, reflects root translation in world X, and
transforms axis-angle axial vectors with the matching sign changes. Standard
AMASS conversion then resamples to20fps and regenerates all205 features. This
preserves consistent rotations, velocities, yaw and local joint features.
The fixed skeleton is not perfectly symmetric, so mirrored amplitudes are
remeasured, not copied from the native left clip. No additional scaling or
retiming is performed. Reference-left caches use task=-1 and their index uses
`sidestep_native_left_reference`, ready_for_training=false; they cannot match the
current sidestep task selector. Native left and mirrored right always share a
source family and split. Current training manifests contain rightward mirrors.

## Coverage and verification

Training directed displacement0.504956–0.934680m, all34 inside0.4–1.2m.
Ten equal-width bin counts: `[0,3,4,10,8,6,3,0,0,0]`; four bins remain empty.
Validation0.489857–0.749862m, all6 inside, from only two recordings. This is not
full-range or independent-subject coverage. All four actors appear in train;
validation uses mm/tr recordings, so do not claim actor-independent evaluation.

`review.py` verifies80 exact source crops, positive/negative direction, physical
admission, current task quantities, all19 untouched tasks, loader compatibility
and split separation. Maximum quantity recomputation discrepancy2.87e-8.
Six meaningful tests cover mirror involution, rotation conjugation/determinants,
left/right direction, forward walking and crossing-foot rejection.
`finalize_audit.py` additionally checks exact mirrored raw inputs, text/PCA/masks,
all20 source-family isolation, and snapshots source code and input hashes.

`review.py` picks one seeded cycle from each of10 distinct TRAIN families;
left panels are native references, right panels are actual training caches.
Videos preserve20fps and fixed cameras. Four sampled poses per video were
inspected; user review is still pending.

## Reproduction

From the saved code snapshot on Betail, use the project virtualenv and installed
Hydra dependency path. Configuration is `config/sidestep_repair.yaml`.

```bash
python -m data_processing.sidestep.inspect_sources
python -m data_processing.sidestep.repair
python -m unittest data_processing.sidestep.test_rules
python -m data_processing.sidestep.review
python -m data_processing.sidestep.finalize_audit
```

The repair refuses to overwrite an existing index; review similarly refuses an
existing review directory. Use a fresh output override for a reproduction.
V1 was an internal candidate; V2 gives reference-left caches a distinct nontraining
task marker. Both produce the same40 accepted mirrored-right clips.

Large arrays and videos remain on Betail and in local outputs_amass. Scripts,
policies, the restored interval list and small audit evidence are retained in
this directory and experiments/sidestep_source_repair_20261010.
