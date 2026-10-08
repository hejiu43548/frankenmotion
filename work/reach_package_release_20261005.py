"""Package frozen artifacts, complete final evidence, and configured-server sources."""
import json,shutil,hashlib,tarfile,ast,subprocess,os
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';B=D/'release';B.mkdir(exist_ok=False)
def cp(src,rel):
 dest=B/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for p in (D/'frozen').rglob('*'):
 if p.is_file():cp(p,Path('weights')/p.relative_to(D/'frozen'))
for p in (R/'outputs_amass/gait_demo_20261005/release/server_snapshot').rglob('*'):
 if p.is_file():cp(p,Path('server_snapshot')/p.relative_to(R/'outputs_amass/gait_demo_20261005/release/server_snapshot'))
for p in (R/'work').glob('reach_*_20261005.py'):
 ast.parse(p.read_text());cp(p,Path('server_snapshot/work')/p.name)
for folder in ['videos','report','visual_review']:
 for p in (D/folder).rglob('*'):
  if p.is_file():cp(p,Path(folder)/p.relative_to(D/folder))
for p in D.rglob('*.json'):
 if any(x in p.parts for x in ['release','videos','report','visual_review','frozen']):continue
 cp(p,Path('evidence')/p.relative_to(D))
for s in (D/'final_paired16').glob('scene_*'):
 for p in s.glob('*.npz'):cp(p,Path('evidence/final_paired16')/s.name/p.name)
 for name in ['selected','selected_cpu']:
  for fn in ['actual.npz','motion.npz']:cp(s/name/fn,Path('evidence/final_paired16')/s.name/name/fn)
for p in (D/'prompts').glob('*'):
 if p.is_file():cp(p,Path('server_snapshot/outputs_amass/reach_demo_20261005/prompts')/p.name)
for p in (D/'training').glob('*/source_snapshot/*.py'):cp(p,Path('evidence/training')/p.relative_to(D/'training'))
for folder in ['training_v7']:
 for s in (D/folder).glob('scene_*'):
  for p in s.glob('*.npz'):cp(p,Path('training_data')/folder/s.name/p.name)
for name in ['scene.mjb','motion.npz','actual.npz','inference_contract.json','result.json','contacts.json','audit.json']:cp(D/'final_paired16/scene_000/selected'/name,Path('example_scene')/name)
cp(D/'final_paired16/scene_000/reference_contact.json',Path('reference_contact.json'))
for name in ['cpu_evaluate','metrics','audit_rollout','render','gait_metrics']:cp(R/'work'/f'reach_{name}_20261005.py',Path('cpu_demo')/f'reach_{name}_20261005.py')
cp(R/'outputs_amass/table_demo_20261005/dependency_inventory.json',Path('existing_generator_dependencies.json'))
cp(D/'sole_geometry.json',Path('server_snapshot/outputs_amass/reach_demo_20261005/sole_geometry.json'))
(B/'README.txt').write_text('''FrankenMotion -> G1: learned reach commands and generated departure, 2026-10-05

Read report/实验说明.txt first. Videos are recorded physical trajectories at 1x speed. A single fixed actor executes approach, reach, lowering and departure. No scene-aware hand IK or per-task tracker selection at execution. Standard GMR IK remains part of retargeting.

Reach command is held wrist forward distance from pelvis, in human-equivalent metres; it is not literal G1 palm travel. See frozen protocol and report for scaling and fixed table-height calibration. Reach and departure adapters are learned components of the diffusion denoiser. Training uses synthetic FK targets; inference does not rewrite sampled hands or legs. Retargeting and clip composition include disclosed root-height grounding, rigid placement, nominal stance blends and temporal resampling.

Run ON THE CONFIGURED SERVER:
cd /home/pku/frankenmotion
work/g1_sim_env/bin/python work/reach_run_demo_20261005.py --name new_reach_demo_001 --seed 88005000 --reach .3 --exit-task sidestep --exit-command .5

Repeat with a new name and --reach .5, keeping the same seed/layout to compare. Optional --distance 1.2 --direction-deg 10 sets approach. All names must be new. This requires the existing server environments and external base-generator assets; this is not a self-contained relocatable training install. Existing dependencies are inventoried. All original intermediate checkpoints remain on the server.

Independent Linux CPU replay from this extracted directory:
python cpu_demo/reach_cpu_evaluate_20261005.py --run example_scene --actor weights/actor.pt --checkpoint weights/policy.pt --output my_cpu_rollout --check-observations
python cpu_demo/reach_audit_rollout_20261005.py --run my_cpu_rollout
python cpu_demo/reach_render_20261005.py --run my_cpu_rollout

Use MuJoCo exactly 3.5.0 for saved MJB; torch 2.7.0, numpy 2.4.6, scipy 1.15.3. Rendering needs EGL/imageio/ffmpeg/Pillow/DejaVu. No CUDA/mjlab needed for this packaged CPU example. Other scene MJBs remain on the server; all final raw traces/references are packaged. Weights include actor.json linking the export to checkpoint SHA.

Evidence retains every final paired test, including failures, plus exploratory metrics. Development results were used for checkpoint selection; final test outcomes were not used for retraining. Selected presentation examples are illustrative, not the success rate. Known table pose, flat ground, fixed 0.8m tabletop and nominal simulation only; no perception, hardware, or general 11-skill retention claim. No automatic push performed.
''')
# Exercise the packaged CPU entry points outside the configured source directory.
check=R/'work/reach_portable_release_check_20261005';check.mkdir(exist_ok=False);shutil.copy2(B/'reference_contact.json',check/'reference_contact.json');env=dict(os.environ);env['PYTHONPATH']='';py=R/'work/mjlab_stable_env/bin/python'
with (check/'execution.log').open('w') as log:
 subprocess.run([str(py),str(B/'cpu_demo/reach_cpu_evaluate_20261005.py'),'--run',str(B/'example_scene'),'--actor',str(B/'weights/actor.pt'),'--checkpoint',str(B/'weights/policy.pt'),'--output',str(check/'rollout'),'--check-observations'],cwd=check,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([str(py),str(B/'cpu_demo/reach_audit_rollout_20261005.py'),'--run',str(check/'rollout')],cwd=check,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
portable=dict(result=json.loads((check/'rollout/result.json').read_text()),parity=json.loads((check/'rollout/observation_parity.json').read_text()),audit=json.loads((check/'rollout/audit.json').read_text()),source='Packaged CPU scripts, empty PYTHONPATH, outside server source directory; same installed Linux runtime')
assert portable['result']['success'];(D/'portable_release_check.json').write_text(json.dumps(portable,indent=2));cp(D/'portable_release_check.json',Path('evidence/portable_release_check.json'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={str(p.relative_to(B)):sha(p) for p in sorted(B.rglob('*')) if p.is_file()};(B/'MANIFEST.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in files.items()));archive=D/'frankenmotion_g1_reach_demo.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:tar.add(B,arcname='frankenmotion_g1_reach_demo')
with tarfile.open(archive,'r:gz') as tar:
 for name,expected in files.items():assert hashlib.sha256(tar.extractfile('frankenmotion_g1_reach_demo/'+name).read()).hexdigest()==expected,name
(D/'release_audit.json').write_text(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files_verified=len(files)),indent=2));print((D/'release_audit.json').read_text())
