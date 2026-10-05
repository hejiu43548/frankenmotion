import json,subprocess,shutil,hashlib,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';OLD=R/'outputs_amass/table_demo_20261005';f=D/'frozen';f.mkdir(exist_ok=False)
for src,name in [(D/'training/distill_v2/model_3000.pt','policy.pt'),(D/'training/distill_v2/actor_3000.pt','actor.pt'),(D/'training/distill_v2/actor_3000.json','actor.json'),(OLD/'frozen/goal_adapter.pt','goal_adapter.pt')]:shutil.copy2(src,f/name)
code=f/'source';code.mkdir()
for pattern in ['gait_*_20261005.py','table_goal_adapter_20261005.py','table_prepare_reference_20261005.py','table_audit_rollout_20261005.py','unified_export_actor_20261004.py']:
 for p in (R/'work').glob(pattern):shutil.copy2(p,code/p.name)
protocol=dict(frozen_unix=time.time(),selected='distill_v2/model_3000: six development CPU and GPU scenes; visual inspection of scene0 and5 walk, scene0 full phases',deployment='One fixed 361-input actor; no routing, no teacher or action mixture at inference',training='SONIC locomotion and previous interaction actor mixed offline only; supervised distillation then DAgger on 8 training references; teacher validation groups14/15 excluded from DAgger',reference='Existing goal-conditioned generator + up to3 same-noise command corrections; uniform GMR; walk-only temporal resampling clip(distance/.45+.8,2.8,4.8); 1.5s settling; explicit scene-aware right-hand lift/extend/lower/hold IK',test=dict(seed=75005000,count=32,all_random=True,range_distance_m=[.85,1.75],range_direction_rad=[-.5,.5],known_table_pose=True,table_top_m=.8),primary_backend='mjlab GPU MuJoCo',independent_check='native CPU MuJoCo; same exported actor and initial-only state writes',gait_diagnostics='knees, foot clearance, contacts and contact-point slip, plus human visual review by assistant; not validated perceptual naturalness metric',limitations=['Arms remain relatively held and stiff','No perception or moving obstacles','Not hardware tested','No new validation of all prior 11 skills'],weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in f.glob('*.pt')})
(f/'protocol.json').write_text(json.dumps(protocol,indent=2))
def run(py,script,args,log):
 with (R/'work'/log).open('w') as o:subprocess.run([str(R/py),str(R/'work'/script)]+args,cwd=R,stdout=o,stderr=subprocess.STDOUT,check=True)
run('.conda/bin/python','gait_generate_scenes_20261005.py',['--weight',str(f/'goal_adapter.pt'),'--name','final_random32','--count','32','--seed','75005000','--all-random','--refine-iterations','3'],'gait_final_generate.log')
folder=D/'final_random32'
run('work/g1_sim_env/bin/python','table_prepare_reference_20261005.py',['--folder',str(folder),'--tag','reference_contact','--lift','--hold-extra','60','--contact-preload','.015'],'gait_final_reference.log')
run('work/mjlab_stable_env/bin/python','gait_retime_20261005.py',['--folder',str(folder)],'gait_final_retime.log')
run('work/g1_sim_env/bin/python','gait_evaluate_20261005.py',['--folder',str(folder),'--checkpoint',str(f/'policy.pt'),'--name','selected','--fresh-initial-observation'],'gait_final_gpu.log')
for row in json.loads((folder/'manifest.json').read_text()):
 s=Path(row['source']);run('work/mjlab_stable_env/bin/python','gait_cpu_evaluate_20261005.py',['--run',str(s/'selected'),'--actor',str(f/'actor.pt'),'--checkpoint',str(f/'policy.pt'),'--output',str(s/'selected_cpu')],f'gait_final_cpu_{row["index"]:03d}.log')
 for name in ['selected','selected_cpu']:
  run('work/mjlab_stable_env/bin/python','gait_metrics_20261005.py',['--run',str(s/name)],f'gait_final_metrics_{row["index"]:03d}_{name}.log')
  run('work/mjlab_stable_env/bin/python','table_audit_rollout_20261005.py',['--run',str(s/name)],f'gait_final_audit_{row["index"]:03d}_{name}.log')
(D/'final_complete.json').write_text(json.dumps(dict(completed_unix=time.time(),count=32)))
