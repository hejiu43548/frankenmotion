# back_walk: crop away forward phases

Scope: only existing back_walk source annotations; all other19 tasks are retained
byte-for-byte as JSON records. No training was started. The base is the actual
latest shared20_clean5_stage2_20261010/data snapshot, not an older repair branch.

`repair.py` reads the existing full20fps source within each original annotation,
locates continuous negative forward velocity relative to the current hip heading,
and crops2–6second windows. It removes padded tails and rejects records without
an admissible backward interval. A world-negative trajectory alone is insufficient:
a person who turns around and walks forward must not become a back_walk label.

Admission also checks positive recomputed task speed, <=2% forward path fraction,
limited lateral travel/heading changes, bilateral ankle stepping and upright torso.
After TRAIN preview showed a bent-over retreat, V2 adds maximum torso tilt0.70rad.
This is a direction/crop repair, not an exhaustive exclusion of every arm gesture
or secondary activity. Thresholds are saved in config/back_walk_repair.yaml.

Original captions can describe forward phases or merely arms moving back/forth.
The new local action/leg and global caption is `walk backwards`; original captions,
keys, old quantities, source annotation times, old caches and new exact crop bounds
are all retained. It is a kinematically verified crop descriptor, not invented
BABEL/native event metadata. No new source family, mirroring, time reversal,
amplitude scaling, retiming or padding is introduced.

Original1118train/135val records include763 nonpositive stored task quantities;
1078 have >2% forward-path fraction on their original real frames. These overlap
and are not two disjoint error counts. Final156train/25val clips come from137/22
source families.159 original records contribute181 windows;1094 originals have
no admitted window under this policy. This does not mean every rejected record
was purely forward walking: duration, turning, lateral motion and posture also
contribute to exclusion. before.json/rejected.json retain the evidence.

All181 saved motions equal their source slices exactly; speed recomputation error0,
forward-path fraction at most0.1985%, all20 family splits remain disjoint, and
text/PCA/masks are checked. Other19 tasks are unchanged. Five regression checks
cover pure-forward rejection, mixed-direction trimming, heading-aware direction,
short/stationary rejection and bent-over rejection.

Run with Hydra from the saved source snapshot:

```bash
python -m data_processing.back_walk.repair
python -m unittest data_processing.back_walk.test_rules
python -m data_processing.back_walk.review
```

Use a fresh output override for reproduction. Large arrays/video live at Betail
outputs_amass/source_repair_back_walk_20261010_v2. Ten seeded TRAIN clips from ten
different families are rendered at20fps; user review is pending. Small reports,
train/val lists and hashes are in experiments/back_walk_source_repair_20261010.
