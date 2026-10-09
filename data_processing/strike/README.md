# Strike: right-forward straight punch/jab

User-confirmed scope: **only right-hand forward straight punches and jabs**.
Hooks, uppercuts, side punches, left punches and mixed kicking actions are excluded.
This changes source selection only, preserving the existing velocity metric.

`repair.py` scans all available BABEL train/val/extra frame annotations for the
exact category `punch`. HDM05 scene03-02 recordings are considered first; their
temporal boundaries currently come from BABEL. The official HDM05 cuts endpoint
returns403, so these are **not claimed to be verified official native cuts**.
Existing Frankenstein source-family splits remain authoritative. BABEL's split
filename is annotation provenance; existing test/unassigned families never enter
the output. No annotation-label regex fallback is used for admission.

The initial investigation found21 object-box caption collisions among123 old
training rows (`\bbox` had matched carrying/placing a box).113 old metric peaks
lacked a matching BABEL punch category. Missing BABEL evidence is an uncertainty
flag, not proof that all113 are wrong.91 lacked fast right-forward motion at the
old metric peak. Flags can overlap; the full old-row audit preserves evidence.

## Rules and timing

`rules.py` defines every threshold. Find local peaks of shoulder-relative right
wrist forward speed inside the annotated punch. Require forward travel, elbow
extension and right-arm dominance. Bound root/foot displacement and raised-foot
motion to exclude locomotion/kicking contamination. Existing thresholds are
explicit conservative engineering choices, not learned contact ground truth.

The legacy quantity uses smoothed world-space right-wrist speed indices14:32,
whose centers are original crop frames16..33. Choose the longest qualifying real
window from60/50/40 frames, trying outbound-peak positions28/24/20/16/32 in that
order. Do not stretch or pad. Reject conflicting neighboring BABEL actions and
verify the metric maximum still measures the selected outbound punch rather than
retraction or another movement. Require no less than50% non-overlap relative to
the shorter same-source window. Store every failed crop attempt.

The first fixed3s draft rejected valid isolated punches because context included
the next side punch/kick. Version2 varies context length/alignment while keeping
the physical thresholds. Version3 additionally excludes physically detected left-forward straight punches anywhere
in the crop, including outside the command measurement window. The final candidate
has13train/1val events across10/1 source families:4 training events from HDM05
recordings and9 from other BABEL sources. Training speeds span1.788–4.375m/s;
the single validation sample is4.862m/s. The configured1.5–2.5m/s command range is
**not fully covered**. One validation source cannot establish generalization.
No samples are duplicated or retimed to fill the range.

Original event text conditions the sequence. Action/right-arm text is present
only within the original BABEL time interval; preparation/reset and unspecified
parts remain unknown. Reuse original CLIP/PCA, encoding only missing actual labels
with the same ViT-B/32 and checking cached-label cosine parity above0.9999.

## Reproduction and audit

```sh
python -m data_processing.strike.repair --out NEW_OUTPUT_DIRECTORY
python -m unittest data_processing.strike.test_rules
python -m data_processing.strike.review NEW_OUTPUT_DIRECTORY
python -m data_processing.export_task_audits \
  --data NEW_OUTPUT_DIRECTORY --out NEW_AUDIT_DIRECTORY \
  --review-state data_processing/review_state.json
```

Use the pinned project environment and CLIP dependencies on `PYTHONPATH`; CLI
arguments expose the source paths. Retain `policy.json`, `provenance.json`,
`sources.json`, `rejected.json`, `old_strike_audit.json`, exact split manifests,
cache hashes and the code snapshot. The review uses seed20261009 and ten distinct
training recordings; fixed-camera source videos are20fps, with optional0.5×
player speed. Other19 task records must remain identical, including provisionally
accepted wave. User provisionally accepted this dataset with known defects on2026-10-09.
Mixed-punch contamination remains according to user review. Continue cleaning kick
before unified training; see `../MEMO.md`. This is not a clean-data certification.
