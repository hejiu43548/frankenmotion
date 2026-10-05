import sys,json,hashlib,subprocess,importlib.metadata as md
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[R/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt',R/'outputs_amass/root_control_20260925/best.pt',R/'outputs_amass/root_control_20260925/base_config.yaml',R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt',R/'outputs_amass/franken_eleven_20261003/skeleton.npz',R/'outputs_amass/franken_eleven_20261003/evaluation_manifest.json',R/'work/beyondmimic_demo_motion.npz']
for folder,pat in [('outputs_amass/franken_eleven_20261003/code','*.py'),('outputs_amass/root_control_20260925/code','*.py'),('outputs_amass/franken_eleven_20261003/prompts','walk_p*.pt'),('outputs_amass/franken_eleven_20261003/prompts','reach_p0.pt'),('outputs_amass/franken_improve_20261003/frozen_retarget_v2','*')]:files.extend(p for p in (R/folder).glob(pat) if p.is_file())
files.extend(R/'work'/n for n in ['physical_adapter_20261003.py','gmr_probe_20261003.py','unified_retarget_20261004.py','unified_preview_20261004.py','unified_export_actor_20261004.py','mjlab_cpu_fk_20261003.py','unified_train_20261004.py','unified_motion_20261004.py','unified_rewards_20261004.py','unified_calibration_20261004.py'])
files.extend(p for p in (R/'work/GMR').rglob('*') if p.is_file() and p.suffix in ['.py','.json','.xml'])
records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
versions={}
for package in ['torch','numpy','scipy','mujoco','mujoco-warp','mjlab','rsl-rl-lib','imageio','imageio-ffmpeg','Pillow','warp-lang']:
 try:versions[package]=md.version(package)
 except md.PackageNotFoundError:versions[package]='not registered in this runtime'
repos={}
for key,path in [('frankenmotion',R),('gmr',R/'work/GMR'),('previous_published_snapshot',R/'work/git_publish_g1')]:
 p=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True);top=subprocess.run(['git','-C',str(path),'rev-parse','--show-toplevel'],capture_output=True,text=True);repos[key]=dict(path=str(path),enclosing_git_root=top.stdout.strip(),head=p.stdout.strip() if p.returncode==0 else None,independent_repository=top.stdout.strip()==str(path))
(D/'dependency_inventory.json').write_text(json.dumps(dict(runtime=sys.executable,python=sys.version,versions=versions,repositories=repos,external_assets=records,scope='Full generation uses the configured server and original assets at these paths. Compact delivery contains new adapters, shared policy and an independently executable CPU scene; it does not redistribute the complete original generator or all training data.'),indent=2));print('Recorded',len(records),'dependencies',versions)
