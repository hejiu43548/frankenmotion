"""One-command demo on the configured server; never pushes code or touches hardware."""
import argparse,json,subprocess,math,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=58005000);p.add_argument('--count',type=int,default=1);p.add_argument('--table-x',type=float);p.add_argument('--table-y',type=float);p.add_argument('--no-render',action='store_true');a=p.parse_args()
assert a.name.replace('_','').replace('-','').isalnum();assert a.count>0;assert not (D/a.name).exists(),'Choose a fresh name'
frozen=json.loads((D/'frozen/protocol.json').read_text());goal=D/'frozen/goal_adapter.pt';tracker=D/'frozen/policy.pt'
for path in [goal,tracker]:assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['weights'][path.name]
args=['--weight',str(goal),'--name',a.name,'--seed',str(a.seed),'--count',str(a.count),'--all-random','--refine-iterations','3']
if a.table_x is not None or a.table_y is not None:
 assert a.table_x is not None and a.table_y is not None and a.count==1
 distance=math.hypot(a.table_x,a.table_y)-.65;angle=math.atan2(a.table_y,a.table_x)
 assert .85<=distance<=1.75 and abs(angle)<=.5,'Table must be within the demonstrated front-sector workspace: stopping distance 0.85–1.75m, direction ±0.5rad. Table centre is 0.65m beyond stopping goal.'
 req=D/'requests';req.mkdir(exist_ok=True);layout=req/(a.name+'.json');assert not layout.exists();layout.write_text(json.dumps([dict(distance_robot_m=distance,direction_rad=angle)]));args+=['--layout-json',str(layout)]
def call(runtime,script,args):subprocess.run([str(R/runtime),str(R/'work'/script),*args],cwd=R,check=True)
call('.conda/bin/python','gait_generate_scenes_20261005.py',args)
call('work/g1_sim_env/bin/python','table_prepare_reference_20261005.py',['--folder',str(D/a.name),'--tag','reference_contact','--lift','--hold-extra','60','--contact-preload','.015'])
call('work/mjlab_stable_env/bin/python','gait_retime_20261005.py',['--folder',str(D/a.name)])
call('work/g1_sim_env/bin/python','gait_evaluate_20261005.py',['--folder',str(D/a.name),'--checkpoint',str(tracker),'--name','unified','--fresh-initial-observation'])
for row in json.loads((D/a.name/'manifest.json').read_text()):
 run=Path(row['source'])/'unified';call('work/mjlab_stable_env/bin/python','table_audit_rollout_20261005.py',['--run',str(run)])
 if not a.no_render:call('work/mjlab_stable_env/bin/python','gait_render_20261005.py',['--run',str(run)])
print('Complete:',D/a.name/'unified_summary.json')
