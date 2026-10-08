"""Audit all 11 quantities from nominal native CPU rollouts, separately from GPU protocol."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
from audit_results import TASKS,RANGES,TOLS,stats
from unified_metrics_20261004 import native_metrics
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();out=R/'outputs_amass/unified_generator_20261007/general_evaluation'/a.name;protocol=json.loads((out/'protocol.json').read_text());rows=json.loads((out/'results.json').read_text());assert len(rows)==protocol['planned'];assert hashlib.sha256(Path(protocol['checkpoint']).read_bytes()).hexdigest()==protocol['checkpoint_sha256']
model=tr.rt.load_model();height=tr.robot_height(model);native=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));source_joint_names=[model.joint(i).name for i in range(1,model.njnt)];qorder=[int(native.jnt_qposadr[native.joint('robot/'+name).id]) for name in source_joint_names];audited=[];raw_hashes={}
for row in rows:
 ref=np.load(row['reference_path'])['reference_qpos'];actual=None
 if 'error' not in row:
  raw=Path(row['run'])/'actual.npz';z=np.load(raw);states=np.c_[z['qpos'][:,:7],z['qpos'][:,qorder]];assert np.isfinite(states).all();raw_hashes[str(raw)]=hashlib.sha256(raw.read_bytes()).hexdigest()
  if row['physical_complete']:
   assert len(states)==row['planned_frames'];ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;assert ts[-1]>=want[-1]-1e-7;positions=tr.get_positions(model,states);positions=np.stack([np.interp(want,ts,v) for v in positions.reshape(len(states),-1).T],1).reshape(-1,24,3);actual=tr.measure(positions,row['task'],height)
 audited.append(dict(row,human=native_metrics(row),g1=tr.measure(tr.get_positions(model,ref),row['task'],height),actual=actual))
summary={}
for i,task in enumerate(TASKS):
 rr=[r for r in audited if r['task']==task]
 if not rr:continue
 span=RANGES[i][1]-RANGES[i][0];summary[task]={k:stats(rr,k,span,TOLS[i]) for k in ['human','g1','actual']}
 for k in ['human','g1','actual']:summary[task][k]['semantic_E_all']=float(np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None and r[k]['event_pass'] else 1. for r in rr]))
 summary[task]['reference_fidelity_E_all']=float(np.mean([min(abs(r['actual']['quantity']-r['g1']['quantity'])/span,1) if r['actual'] is not None else 1. for r in rr]))
aggregate=dict(requests=len(rows),complete=sum(v['actual']['measurable'] for v in summary.values()),event=sum(v['actual']['event_pass'] for v in summary.values()),joint=sum(v['actual']['joint_pass'] for v in summary.values()),macro_semantic_E_all=float(np.mean([v['actual']['semantic_E_all'] for v in summary.values()])),macro_reference_fidelity_E_all=float(np.mean([v['reference_fidelity_E_all'] for v in summary.values()])))
(out/'audited_results.json').write_text(json.dumps(audited,indent=2,default=lambda x:x.item()));(out/'eleven_summary.json').write_text(json.dumps(summary,indent=2));(out/'eleven_audit.json').write_text(json.dumps(dict(aggregate=aggregate,checkpoint_sha256=protocol['checkpoint_sha256'],backend='native CPU nominal model; report separately from GPU randomized protocol',raw_trajectory_hashes=raw_hashes),indent=2));print(json.dumps(aggregate),flush=True)
