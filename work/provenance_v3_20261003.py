"""Hash frozen experimental assets and preserve code/environment provenance."""
from pathlib import Path
import json,hashlib,subprocess,datetime
ROOT=Path('/home/pku/frankenmotion');OUT=ROOT/'outputs_amass/franken_improve_20261003';BASE=ROOT/'outputs_amass/franken_eleven_20261003'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def git(path):
 p=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True);return p.stdout.strip() if p.returncode==0 else None
files=[ROOT/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt',ROOT/'outputs_amass/root_control_20260925/best.pt',BASE/'task_adapter_deploy.pt',OUT/'frozen_generation_v1/physical.pt',OUT/'frozen_generation_v1/jump_aligned.pt',ROOT/'work/beyondmimic_demo.pt',ROOT/'work/beyondmimic_demo_motion.npz']
files+=list((ROOT/'work/g1_sonic_official/gear_sonic_deploy/policy/sonic_v1_1').glob('*.onnx'))
files += [BASE/'skeleton.npz',BASE/'evaluation_manifest.json',BASE/'data_manifest.json',ROOT/'outputs_amass/root_control_20260925/base_config.yaml',ROOT/'work/g1_sonic_official/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml']
files += list((BASE/'prompts').glob('*.pt'))
files += list((OUT/'frozen_controllers_v1').glob('*'))
files += [OUT/'frozen_generation_v1/protocol.json']
for folder in ['frozen_retarget_v2','frozen_kick_v3','frozen_integrated_v3']:files += [p for p in (OUT/folder).glob('*') if p.is_file()]
for folder in ['beyondmimic_finetune_turn_all_ori2.0_vel1.0','beyondmimic_finetune_strike_all_ori0.5_vel2.0']:
 files += [OUT/folder/'model_2399.pt',OUT/folder/'protocol.json']
files += [OUT/'specialist_selection_rule.json',OUT/'capacity_correction_protocol.json']
record=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git={'frankenmotion':git(ROOT),'sonic':git(ROOT/'work/g1_sonic_official'),'GMR':git(ROOT/'work/GMR'),'GMR_pinned_source_commit':'bb1bbe40774794fceb2a7c579a3464a28e68c844','mjlab_pinned_tag_commit':'3cb20cb64507f4d1cf5c8f271ff79b5144727025'},assets={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files if p.exists()},experiment_code={str(p):sha(p) for p in (ROOT/'work').glob('*_20261003.py')},base_experiment_code={str(p):sha(p) for p in (BASE/'code').glob('*.py')},environments={name:json.loads((OUT/(name+'_environment.json')).read_text()) for name in ['generator','sonic','mjlab']})
(OUT/'asset_provenance_v3.json').write_text(json.dumps(record,indent=2));print('Hashed',len(record['assets']),'assets,',len(record['experiment_code']),'scripts')
