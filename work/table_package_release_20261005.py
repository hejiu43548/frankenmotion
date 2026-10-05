import json,hashlib,shutil,tarfile,ast
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';B=D/'release';B.mkdir(exist_ok=False)
def copy(src,dst):
 dst=B/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for p in (R/'work').glob('table_*_20261005.py'):ast.parse(p.read_text());copy(p,Path('server_snapshot/work')/p.name)
inv=json.loads((D/'dependency_inventory.json').read_text())
for row in inv['external_assets']:
 p=Path(row['path'])
 if p.suffix in ['.py','.json','.yaml','.xml'] or p.name=='skeleton.npz':copy(p,Path('server_snapshot')/p.relative_to(R))
for p in (D/'frozen').iterdir():
 if p.is_file():copy(p,Path('weights')/p.name)
copy(R/'outputs_amass/franken_unified_20261004/frozen_unified/policy.pt',Path('weights/original_policy.pt'))
for p in (D/'report').iterdir():copy(p,Path('report')/p.name)
for p in (D/'videos').iterdir():
 if p.is_file():copy(p,Path('videos')/p.name)
for rel in ['command_sweep/direction_distance_demo.mp4','command_sweep/video_metadata.json','command_sweep/grid_frame_436.png','cpu_final/presentation.mp4','cpu_final/presentation.png']:
 copy(D/rel,Path('extra_demos')/rel)
# All experiment JSON evidence includes exploratory failures and selection tradeoffs.
for p in D.rglob('*.json'):
 if B in p.parents or any(x in p.parts for x in ['videos','report','frozen']):continue
 copy(p,Path('evidence')/p.relative_to(D))
for p in (D/'goal_command_audit').iterdir():
 if p.suffix in ['.npz','.csv','.png','.pdf']:copy(p,Path('evidence/goal_command_audit')/p.name)
for scene in (D/'final_test').glob('scene_*'):
 for p in scene.glob('*.npz'):copy(p,Path('evidence/final_test')/scene.name/p.name)
 for method in ['unified','original_tracker']:
  for name in ['actual.npz','motion.npz']:copy(scene/method/name,Path('evidence/final_test')/scene.name/method/name)
# Exact native model plus complete example for independently executing the final actor.
hero=D/'final_test/scene_000/unified'
for name in ['scene.mjb','actual.npz','motion.npz','inference_contract.json','result.json','contacts.json','audit.json']:copy(hero/name,Path('example_scene')/name)
for name in ['cpu_controller','audit_rollout','render_presentation']:copy(R/f'work/table_{name}_20261005.py',Path('cpu_demo')/(name+'.py'))
readme='''# FrankenMotion goal → G1 table contact (2026-10-05)

This is a locally saved research release. Nothing was pushed. Read `report/实验说明.txt` for results, selection history, limitations and next steps. Videos depict recorded physical states at 1× speed. No physical state was overwritten after initialization during execution.

## Contents

- `videos/funding_demo.mp4`: intro, full first successful final trial, first four random trials (all retained), results.
- `extra_demos/command_sweep/direction_distance_demo.mp4`: same-noise direction/distance demonstrations.
- `weights/goal_adapter.pt`: new learned goal branch ONLY; the original generator is an external dependency.
- `weights/policy.pt`: selected shared full-body tracker, retention_v2 update 500.
- `weights/actor.pt`: exported TorchScript actor with normalization; one actor for all phases/scenes.
- `weights/original_policy.pt`: previous shared tracker for the paired comparison.
- `weights/protocol.json`: freeze time, hashes, selection and exact pipeline.
- `server_snapshot/`: exact new experiment scripts and referenced code/configuration. Scripts target the existing configured server paths, not an automatically relocated installation.
- `evidence/`: raw metrics/protocols (including exploratory failures), final 64 state/control traces and generated references. Full native models for every rollout remain on the server; one exact model is bundled below.
- `example_scene/`: first final random scene with exact MuJoCo 3.5.0 binary, recorded observations/actions, reference, initialization and metric contract.
- `cpu_demo/`: standalone CPU execution and audit, no CUDA or mjlab dependency.
- `MANIFEST.sha256`: file content checksums for this release.

## Run a NEW generated table scene on the configured server

```bash
cd /home/pku/frankenmotion
work/g1_sim_env/bin/python work/table_run_demo_20261005.py --name new_random_table_001 --seed 58006001 --count 1
```

`--name` must not already exist. Optional `--table-x 1.9 --table-y 0` sets tabletop centre XY (count must be 1). Supported stopping radius is 0.85–1.75 m and direction ±0.5 rad; table centre is another 0.65 m away. Coordinates are known simulator coordinates. The command generates, refines input commands at most three times, runs GMR and scene-aware IK, executes the frozen policy, audits contact, and renders a new video. It does not select a different tracker by task. It requires the existing FrankenMotion/retarget/runtime assets recorded in `evidence/dependency_inventory.json`.

## Execute the packaged example on Linux CPU

Tested with Python 3.11, torch 2.7.0, numpy 2.4.6, scipy 1.15.3 and mujoco 3.5.0. Use a compatible environment; the exact scene binary is version-specific. No Mac or real-robot compatibility claim is made.

From this extracted release directory:

```bash
python cpu_demo/cpu_controller.py --run example_scene --actor weights/actor.pt --output my_cpu_rollout --check-observations
python cpu_demo/audit_rollout.py --run my_cpu_rollout
```

The output directory must be new. This builds observations, runs the actor, sends joint targets and calls `mj_step`; it does not replay qpos. The recorded reference is the controller's target, not an injected robot state. CPU/GPU numerical trajectories can differ slightly. Optional EGL rendering (imageio, imageio-ffmpeg, Pillow, DejaVu fonts required):

```bash
python cpu_demo/render_presentation.py --run my_cpu_rollout --label 'Independent CPU execution'
```

The actor expects the exact observation ordering/action scaling in `example_scene/inference_contract.json`; it is not a plug-and-play hardware controller.

## Evidence interpretation

Final 32 layouts and noise seeds were sampled after selection/freeze. All have known table pose, fixed 0.8 m table height and fixed walk/reach language templates. No final case was removed. Evaluation uses nominal physics; this is not a domain-randomized robustness score. Success combines stable completion, root error <0.2 m, palm-centre target error <0.1 m and ≥1 s consecutive 50 Hz sampled top-contact with force >0.2 N. Full JSON exposes each component.

New goal conditioning is learned, but the whole interaction is not end-to-end learned: numerical input-command refinement, GMR, smooth standing transition and explicit lift/extend/lower IK are part of the planner. Generation-only raw ablation is separate and does not use that refinement. Some old scalar speed conditions extrapolate beyond the old adapter's validated range, so the ablation is not a general ranking of the old model.

Prior shared tracker also completes 31/32 with the same new planner. The new tracker primarily improves endpoint precision; its old 110-request joint-pass count is 47 versus 49 previously. This tradeoff is preserved rather than hidden.

Full training checkpoints and all native models remain at `/home/pku/frankenmotion/outputs_amass/table_demo_20261005`. Training protocols, source SHA256 and candidate results are in evidence; the compact archive does not redistribute all old datasets, environments or original generator weights. No push or hardware operation was performed.
'''
(B/'README.md').write_text(readme)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
files=sorted(p for p in B.rglob('*') if p.is_file());manifest={str(p.relative_to(B)):sha(p) for p in files};(B/'MANIFEST.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in manifest.items()))
archive=D/'frankenmotion_g1_table_demo.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:tar.add(B,arcname='frankenmotion_g1_table_demo')
# Check every archived byte against the manifest, without trusting file existence alone.
with tarfile.open(archive,'r:gz') as tar:
 for rel,expected in manifest.items():
  h=hashlib.sha256();f=tar.extractfile('frankenmotion_g1_table_demo/'+rel)
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  assert h.hexdigest()==expected,rel
(D/'release_audit.json').write_text(json.dumps(dict(files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),all_archive_entries_verified=True,code_pushed=False),indent=2));print((D/'release_audit.json').read_text())
