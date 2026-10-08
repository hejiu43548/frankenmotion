"""Frozen SONIC release mode0 in its existing native PD/model deployment bridge.
This is a system-level native-deployment baseline, not a matched actuator ablation.
"""
import os,sys,json,hashlib,argparse,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work/g1_sim_bridge'),str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import g1_runtime as rt
import transfer as tr
from audit_results import TASKS,RANGES,TOLS,stats
from unified_metrics_20261004 import native_metrics
G={}
def init(out):
 m=rt.load_model();c=json.loads((D/'evaluation/baseline_broad_dev/contract.json').read_text());names=[m.joint(i).name for i in range(1,m.njnt)];G.update(m=m,p=rt.Policy(),out=Path(out),order=[c['joint_names'].index(n) for n in names],height=tr.robot_height(m))
def one(row):
 start=time.monotonic();m,p=G['m'],G['p'];z=np.load(row['motion_path']);q=z['joint_pos'][:,G['order']];dq=z['joint_vel'][:,G['order']];quat=z['body_quat_w'][:,0];d=mujoco.MjData(m);d.qpos[:7]=np.r_[z['body_pos_w'][0,0],quat[0]];d.qpos[7:]=rt.Q0;rt.floor_align(m,d);p.reset(d);states=[d.qpos.copy()];vel=[d.qvel.copy()];actions=[];ctrl=[];fall=None
 for i in range(len(q)-1):
  target,act=p.act(d,q,dq,quat,i)
  for _ in range(10):
   d.ctrl[:]=np.clip(rt.KP*(target-d.qpos[7:])-rt.KD*d.qvel[6:],-rt.EFF,rt.EFF);mujoco.mj_step(m,d)
  mujoco.mj_forward(m,d);p.update(d,act);states.append(d.qpos.copy());vel.append(d.qvel.copy());actions.append(act.copy());ctrl.append(d.ctrl.copy());w,x,y,zq=d.qpos[3:7];roll=np.arctan2(2*(w*x+y*zq),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-zq*x),-1,1))
  if not np.isfinite(d.qpos).all() or d.qpos[2]<.35 or np.hypot(roll,pitch)>np.pi/3:fall=(i+1)*.02;break
 dest=G['out']/Path(row['path']).stem;dest.mkdir();raw=dest/'actual.npz';states=np.asarray(states);np.savez_compressed(raw,qpos=states,qvel=vel,actions=actions,ctrl=ctrl,fps=50.);ref=np.load(row['reference_path'])['reference_qpos'];actual=None
 if fall is None:
  ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;assert ts[-1]>=want[-1]-1e-7;pos=tr.get_positions(m,states);pos=np.stack([np.interp(want,ts,v) for v in pos.reshape(len(pos),-1).T],1).reshape(-1,24,3);actual=tr.measure(pos,row['task'],G['height'])
 return dict(row,human=native_metrics(row),g1=tr.measure(tr.get_positions(m,ref),row['task'],G['height']),actual=actual,physical_complete=fall is None,termination_time=fall,run=str(dest),wall_s=time.monotonic()-start,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--name',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args();out=D/'sonic_evaluation'/a.name;out.mkdir(parents=True,exist_ok=False);rows=json.loads(Path(a.manifest).read_text());rows=rows[:a.limit] if a.limit else rows
 hashes={n:hashlib.sha256((rt.POLICY/n).read_bytes()).hexdigest() for n in ['model_encoder.onnx','model_decoder.onnx']};protocol=dict(scope=__doc__,manifest=a.manifest,manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),requests=len(rows),policy_hashes=hashes,bridge_sha256=hashlib.sha256(Path(rt.__file__).read_bytes()).hexdigest(),xml=str(rt.XML),xml_sha256=hashlib.sha256(rt.XML.read_bytes()).hexdigest(),mode=0,root_translation_input=False,preview_offsets=list(range(0,50,5)),control_hz=50,physics_hz=500,kp=rt.KP.tolist(),kd=rt.KD.tolist(),effort=rt.EFF.tolist(),entry='same1s reference standing entry,source-model floor alignment,zero velocity',termination='pelvis<.35 or tilt>60deg or nonfinite',caution='Native model/actuator contract differs from BeyondMimic-derived policy. Do not attribute every difference exclusively to network architecture or claim official reproduced paper scores. Existing bridge implements released ONNX mode0.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
 with ProcessPoolExecutor(a.workers,initializer=init,initargs=(str(out),)) as pool:
  for row in pool.map(one,rows):
   results.append(row);(out/'audited_results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()));print(len(results),row['task'],row['physical_complete'],round(row['wall_s'],2),flush=True)
 if a.limit:sys.exit(0)
 summary={}
 for i,task in enumerate(TASKS):
  rr=[r for r in results if r['task']==task];assert len(rr)==len(rows)//11;span=RANGES[i][1]-RANGES[i][0];summary[task]={k:stats(rr,k,span,TOLS[i]) for k in ['human','g1','actual']}
  for k in ['human','g1','actual']:summary[task][k]['semantic_E_all']=float(np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None and r[k]['event_pass'] else 1. for r in rr]))
 aggregate=dict(requests=len(rows),complete=sum(v['actual']['measurable'] for v in summary.values()),event=sum(v['actual']['event_pass'] for v in summary.values()),joint=sum(v['actual']['joint_pass'] for v in summary.values()),macro_semantic_E_all=float(np.mean([v['actual']['semantic_E_all'] for v in summary.values()])))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audit.json').write_text(json.dumps(dict(aggregate=aggregate,policy_hashes=hashes),indent=2));print(aggregate)
