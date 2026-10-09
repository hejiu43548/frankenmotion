# Auditable motion-source processing

Keep task selection, temporal cropping, label reconstruction, admission checks,
and review rendering here. Processing never launches training. Each repair writes
a new dataset directory and preserves every other task's records exactly.

Current scope (2026-10-10): **sidestep only**, using HDM05 non-crossing
walkLeft phases and rightward mirrors for the existing right-only quantity.
All other tasks are deferred by the user. Wave and strike remain provisionally
accepted; kick is not silently marked approved by the request to move on.
The wave/strike/kick stage2 diagnostics are complete. Sidestep was provisionally
accepted on2026-10-10 and its single-task stage2 diagnostic is now authorized.
Each repaired task requires10 rendered training samples for user review;
`review_state.json` records the current scope and gate.

`export_task_audits.py` exports all20 tasks separately as exact train/val JSON,
human-readable CSV and an HTML index, preserving crop bounds, quantities, captions,
source families, split membership, cache hashes and task review status. Exporting
an unreviewed task's list does not certify its semantics.

Prefer native dataset classes and documented event boundaries. For HDM05, record
the mapping from official cut frames to the actual AMASS file/time base. A scene
name alone is not a class or a crop. When published cuts are absent, mark any
supplemental boundaries as such and retain their evidence. For BABEL, use frame
labels with `act_cat`; sequence-only labels and regex hits do not establish an
event boundary. Regex may refine candidate semantics but never overrides a
conflicting native/category label. Always recompute control quantities on the
actual crop. Native class coverage does not imply coverage of parameter ranges.

`wave/` contains the first repair. Future task implementations belong in their own
subdirectories. Do not rewrite historical training manifests or evidence.

Required output evidence:

- Source/annotation identifiers, original labels, split membership and input hashes.
- Full admission policy, units, thresholds, source priority and code snapshot.
- Accepted events and rejected candidates with all failed criteria.
- Exact source-frame bounds, cache hashes and independently recomputed quantities.
- Deterministic review seed, sample IDs, video hashes, FPS/frame checks and review status.
- Version notes distinguishing diagnostic attempts, source supplementation and user acceptance.

Large motion arrays, embeddings and media remain in `outputs_amass/` on Betail;
source code, thresholds and small supplemental annotation tables live here.
