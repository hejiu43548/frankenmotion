"""Reproduce a new full physical demo using frozen generator and one tracker."""
import argparse,json,subprocess,math,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=88005000);p.add_argument('--reach',type=float,default=.3);p.add_argument('--exit-task',choices=['back_walk','sidestep','none'],default='back_walk');p.add_argument('--exit-command',type=float);p.add_argument('--distance',type=float);p.add_argument('--direction-deg',type=float,default=0.);a=p.parse_args();assert '/' not in a.name and not (D/a.name).exists();f=D/'frozen';protocol=json.loads((f/'protocol.json').read_text())
for name,expected in protocol['weights'].items():assert hashlib.sha256((f/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt').read_bytes()).hexdigest()==protocol['weights']['goal_adapter.pt']
def run(py,script,args):subprocess.run([str(R/py),str(R/'work'/script)]+args,cwd=R,check=True)
extra=[]
if a.distance is not None:
 assert .85<=a.distance<=1.75 and abs(math.radians(a.direction_deg))<=.5
 layout=D/(a.name+'_layout.json');layout.write_text(json.dumps([dict(distance_robot_m=a.distance,direction_rad=math.radians(a.direction_deg))]));extra=['--layout-json',str(layout)]
run('.conda/bin/python','reach_generate_walk_20261005.py',['--weight',str(f/'goal_adapter.pt'),'--name',a.name+'_walk','--count','1','--seed',str(a.seed),'--all-random','--refine-iterations','3']+extra)
walk=D/(a.name+'_walk');run('work/g1_sim_env/bin/python','reach_prepare_walk_20261005.py',['--folder',str(walk)])
run('.conda/bin/python','reach_generate_custom_v2_20261005.py',['--name',a.name,'--walk-source',str(walk/'scene_000'),'--reach',str(a.reach),'--exit-task',a.exit_task,'--seed',str(a.seed+1000)]+(['--exit-command',str(a.exit_command)] if a.exit_command is not None else []));folder=D/a.name
run('work/g1_sim_env/bin/python','reach_prepare_corpus_v3_20261005.py',['--folder',str(folder)])
run('work/g1_sim_env/bin/python','reach_evaluate_20261005.py',['--folder',str(folder),'--checkpoint',str(f/'policy.pt'),'--name','selected','--fresh-initial-observation'])
s=folder/'scene_000';run('work/mjlab_stable_env/bin/python','reach_cpu_evaluate_20261005.py',['--run',str(s/'selected'),'--actor',str(f/'actor.pt'),'--checkpoint',str(f/'policy.pt'),'--output',str(s/'selected_cpu'),'--check-observations'])
run('work/mjlab_stable_env/bin/python','reach_audit_rollout_20261005.py',['--run',str(s/'selected')]);run('work/mjlab_stable_env/bin/python','reach_render_20261005.py',['--run',str(s/'selected'),'--output',str(folder/'demo.mp4')]);print(folder/'demo.mp4')
