# Standing, right-compatible wave

Status: user provisionally accepted the10-source review ("先暂定吧"). Current
candidate has45train/3val samples. No training is authorized by this pipeline.

## Implementation and execution

- `repair.py`: native HDM05 supplemental events, BABEL frame-category selection,
  overlap rejection, physical admission, source deduplication and new training caches.
- `hdm05_events.json`: conservative interior event bounds supplemented from the
  official scene05-01 script and training-source FK diagnostics. These are **not
  published official cut boundaries**. Native filename/content mismatch and test
  sources are explicitly excluded.
- `rules.py`: measured displacement descriptors and archived baseline rule helpers.
  Current thresholds are `ADMISSION` in `repair.py`; the complete resolved policy is
  exported to each output's `policy.json`.
- `inspect_hdm05.py`: full-sequence training-only joint trajectories and plots.
- `review.py`: exact cache/source and quantity verification; deterministic ten-source
  training sample selection; fixed-camera two-view skeleton videos and HTML index.
- `test_rules.py`: regressions for source identity, crop duration and actual lateral
  oscillation versus a single reach/raise or stationary hand.
- `archive/`: earlier regex-only diagnostic draft. It is superseded and is not an
  accepted training dataset. Historical executable context is recorded remotely.

From the project root, using the existing project Python with Hydra dependencies
and OpenAI CLIP available on `PYTHONPATH`:

```sh
python -m data_processing.wave.repair --out NEW_OUTPUT_DIRECTORY
python -m unittest data_processing.wave.test_rules
python -m data_processing.wave.review NEW_OUTPUT_DIRECTORY
```

`--base`, `--project`, `--babel` and `--clip-checkpoint` override the explicit
Betail defaults. Output directories must not already exist. Only missing original
event texts are encoded; the cached CLIP vectors and PCA are reused. Newly encoded
labels are saved with parity checks against three known cached labels.

## Admission and provenance

The task's existing quantity measures right-wrist lateral amplitude. Right-only
and both-arm waving can qualify; left-only waves cannot supervise that quantity.
Require2–6 seconds of real motion without padding, a raised active right wrist,
at least two lateral extrema with4cm prominence, and at least3cm lateral half
range. Keep native measured quantities even outside the configured0.08–0.22m
request range; report actual coverage rather than stretching motions or labels.

Limit root excursion to30cm, mean root speed to0.35m/s, each ankle/toe excursion
to15cm, each ankle/toe height range to12cm, and root height range to15cm. These are
conservative admission thresholds, not ground-truth contact labels. Initial
0.12m/s root filtering wrongly rejected normal planted-foot body sway in native
TRAIN examples; the revised policy constrains the feet more tightly. Validation
examples were not inspected to choose thresholds.

Initial ten-source TRAIN review exposed a BABEL wave segment ending in a deep
forward bend. The final admission additionally limits maximum torso tilt from
vertical to0.6rad. Its rejected source and initial rendering remain in the first
review diagnostic output. The final sample is redrawn using the same seed.

BABEL frame labels must contain the exact category `wave`; accept only the
explicit compatible-category set. Remove overlaps with conflicting or unknown
frame categories, including transitions, T-poses, walking and clapping. Do not
fall back to sequence-only descriptions. Supplemental native HDM05 events take
priority over BABEL windows overlapping at least50% of the shorter crop.

The existing Frankenstein **source-family split remains authoritative across all
20 tasks**. BABEL's train/val filenames are label provenance, not a new training
split; differences are recorded on each row. Current test and unassigned families
are never admitted. This is not an evaluation under the official BABEL split.

Use the original event text as global conditioning and as the action/active-arm
label; unspecified parts remain unknown/masked. This intentionally replaces noisy
whole-take captions and avoids copying unrelated body-part labels. No mirroring,
stabilization, motion generation or synthetic prompt templates are used.

Review sampling is seeded20261009: five native HDM05 and five BABEL samples, each
from a different existing training source; shortages are filled from other
distinct training sources and recorded. Videos are actual training-cache FK at
20fps, with fixed cameras and foot/root paths; they are not skinned meshes.

## References

- [HDM05 official documentation, scene5-1 on PDF page12](https://digital-health-bonn.de/wp-content/uploads/2024/03/cg-2007-2.pdf#page=12).
- [HDM05 official classification cuts](https://resources.mpi-inf.mpg.de/HDM05/cuts/index.html)
  returned403 during this run; no missing wave cut boundary is invented.
- [BABEL official repository](https://github.com/abhinanda-punnakkal/BABEL): sequence
  labels and overlapping frame labels are different annotation levels.
- OpenAI CLIP source commit`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`,
  `ftfy==6.3.1`, `wcwidth==0.2.13`; model checkpoint and annotation hashes are
  recorded per output.
