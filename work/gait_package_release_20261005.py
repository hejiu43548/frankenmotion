import json,shutil,hashlib,tarfile,ast
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';OLD=R/'outputs_amass/table_demo_20261005';B=D/'release';B.mkdir(exist_ok=False)
def cp(src,rel):
 dest=B/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for p in (D/'frozen').rglob('*'):
 if p.is_file():cp(p,Path('weights')/p.relative_to(D/'frozen'))
for p in (R/'work').glob('gait_*_20261005.py'):ast.parse(p.read_text());cp(p,Path('server_snapshot/work')/p.name)
for p in (OLD/'release/server_snapshot').rglob('*'):
 if p.is_file():cp(p,Path('server_snapshot')/p.relative_to(OLD/'release/server_snapshot'))
for folder in ['videos','report','visual_review']:
 for p in (D/folder).glob('*'):
  if p.is_file():cp(p,Path(folder)/p.name)
for p in D.rglob('*.json'):
 if any(x in p.parts for x in ['release','videos','report','visual_review','frozen']):continue
 cp(p,Path('evidence')/p.relative_to(D))
for s in (D/'final_random32').glob('scene_*'):
 for p in s.glob('*.npz'):cp(p,Path('evidence/final_random32')/s.name/p.name)
 for name in ['selected','selected_cpu']:
  for fn in ['actual.npz','motion.npz']:cp(s/name/fn,Path('evidence/final_random32')/s.name/name/fn)
for folder in ['teacher_data','teacher_data_dagger1','teacher_data_combined']:cp(D/folder/'dataset.npz',Path('training_data')/folder/'dataset.npz')
for name in ['scene.mjb','motion.npz','actual.npz','inference_contract.json','result.json','contacts.json','audit.json']:cp(D/'final_random32/scene_000/selected'/name,Path('example_scene')/name)
cp(D/'final_random32/scene_000/reference_contact.json',Path('reference_contact.json'))
for name,src in [('cpu_evaluate.py','gait_cpu_evaluate_20261005.py'),('audit_rollout.py','table_audit_rollout_20261005.py'),('render.py','gait_render_20261005.py')]:cp(R/'work'/src,Path('cpu_demo')/name)
cp(OLD/'dependency_inventory.json',Path('existing_generator_dependencies.json'))
readme='''FrankenMotion → G1: revised walking and table-contact demo, 2026-10-05

Read report/实验说明.txt for results and limitations. No push or hardware operation was performed. This release does not replace the previous release.

Deployment: ONE 361-input fixed actor, weights/actor.pt. No task router, no weight interpolation, no teacher at inference. SONIC and the previous interaction actor were used only to label physical training rollouts, followed by supervised distillation and DAgger. Existing goal adapter is unchanged. Generator, GMR, reference temporal retiming, explicit scene-aware hand IK and tracker are separate stages, not an end-to-end learned contact policy.

Videos are actual recorded physical states at 1x real time. Renderer reconstructs saved states only for drawing; evaluation never injects reference states after initialization. gait_before_after.mp4 compares the same development scene and generated path. The faster pipeline explicitly retimes the reference before physical execution; when its walking clip ends the comparison holds its last frame and labels this.

Run a new scene ON THE CONFIGURED SERVER:
cd /home/pku/frankenmotion
work/g1_sim_env/bin/python work/gait_run_demo_20261005.py --name new_gait_demo_001 --seed 76005000 --count 1

Name must be new. Optional --table-x and --table-y set table centre; supported stopping distance .85–1.75m, direction ±.5rad. Table centre is .65m beyond the stopping goal. Known simulator pose and fixed .8m tabletop. Existing environments/generator assets remain external; scripts use configured absolute server paths. Dependencies are inventoried in existing_generator_dependencies.json and the source snapshot. This is not a relocatable training environment.

Independent Linux CPU execution of packaged example, from this extracted directory:
python cpu_demo/cpu_evaluate.py --run example_scene --actor weights/actor.pt --checkpoint weights/policy.pt --output my_cpu_rollout --check-observations
python cpu_demo/audit_rollout.py --run my_cpu_rollout
python cpu_demo/render.py --run my_cpu_rollout

Use MuJoCo EXACTLY3.5.0 for scene.mjb, torch2.7.0, numpy2.4.6, scipy1.15.3. Rendering additionally needs EGL/imageio/ffmpeg/Pillow/DejaVu fonts. Output folder must be new. No CUDA/mjlab is required for this example. No Mac/hardware compatibility claim. Source observation/action contract is in example_scene/inference_contract.json.

Evidence contains all32 final scenes in both GPU and CPU backends, raw state/control/observation traces, generated references, failures from exploratory versions, and teacher training datasets. One exact87MB physics model is bundled; all per-scene models and intermediate training checkpoints remain on the server at /home/pku/frankenmotion/outputs_amass/gait_demo_20261005. All final cases use one checkpoint and are retained. No new evaluation of the previous11-skill benchmark was conducted; table-demo success is not proof of general skill retention. Nominal physics only, no hardware or perception validation.
'''
(B/'README.txt').write_text(readme)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={str(p.relative_to(B)):sha(p) for p in sorted(B.rglob('*')) if p.is_file()};(B/'MANIFEST.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in files.items()));archive=D/'frankenmotion_g1_gait_demo.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:tar.add(B,arcname='frankenmotion_g1_gait_demo')
with tarfile.open(archive,'r:gz') as tar:
 for name,expected in files.items():assert hashlib.sha256(tar.extractfile('frankenmotion_g1_gait_demo/'+name).read()).hexdigest()==expected,name
(D/'release_audit.json').write_text(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files_verified=len(files)),indent=2));print((D/'release_audit.json').read_text())
