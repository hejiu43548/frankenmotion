from pathlib import Path
import json,sys,argparse,hashlib
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
from audit_results import TASKS,RANGES,TOLS
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();out=D/'general_evaluation'/a.name;rows=json.loads((out/'results.json').read_text());model=tr.rt.load_model();height=tr.robot_height(model);native=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));order=[int(native.jnt_qposadr[native.joint('robot/'+model.joint(i).name).id]) for i in range(1,model.njnt)];audited=[]
for r in rows:
 metric=None
 if r.get('physical_complete'):
  z=np.load(Path(r['run'])/'actual.npz');states=np.c_[z['qpos'][:,:7],z['qpos'][:,order]];positions=tr.get_positions(model,states);n=len(np.load(r['path'])['motion']);want=1+np.arange(n)*.05;ts=np.arange(len(positions))*.02;assert want[-1]<=ts[-1]+1e-7;p=np.stack([np.interp(want,ts,v) for v in positions.reshape(len(positions),-1).T],1).reshape(n,24,3);metric=tr.measure(p,r['task'],height) if r['kind'] in TASKS+['back_departure','side_departure'] else {}
 audited.append(dict(r,actual=metric))
summary={}
for k in sorted(set(r['kind'] for r in rows)):
 rr=[r for r in audited if r['kind']==k];ok=[r for r in rr if r['actual'] is not None];s=dict(count=len(rr),complete=len(ok),mean_joint_rmse_rad=float(np.mean([r['joint_rmse_rad'] for r in ok])) if ok else None,mean_global_mpjpe_m=float(np.mean([r['global_mpjpe_m'] for r in ok])) if ok else None,mean_root_error_m=float(np.mean([r['root_translation_error_m'] for r in ok])) if ok else None)
 if k in TASKS or k in ['back_departure','side_departure']:
  tid=TASKS.index(rr[0]['task']);span=RANGES[tid][1]-RANGES[tid][0];s.update(joint_pass=sum(bool(r['actual'] is not None and r['actual']['event_pass'] and abs(r['actual']['quantity']-r['command'])<=TOLS[tid]) for r in rr),event_pass=sum(bool(r['actual'] is not None and r['actual']['event_pass']) for r in rr),semantic_E=float(np.mean([min(abs(r['actual']['quantity']-r['command'])/span,1) if r['actual'] is not None and r['actual']['event_pass'] else 1 for r in rr])))
 summary[k]=s
(out/'audited_results.json').write_text(json.dumps(audited,indent=2,default=lambda x:x.item()));(out/'assessment.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(requests=len(rows),complete=sum(s['complete'] for s in summary.values()),standard_joint=sum(summary[t]['joint_pass'] for t in TASKS),standard_E=float(np.mean([summary[t]['semantic_E'] for t in TASKS])))))
