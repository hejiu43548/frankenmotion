"""Audit reference-state initialization for geometric interpenetration.
The reference trajectory is never changed. Geometric validity is not stability.
"""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';G={}
def initialize():
 p=D/'table_evaluation/baseline_stable/scene_000/reference_input';m=mujoco.MjModel.from_binary_path(str(p/'scene.mjb'));c=json.loads((p/'inference_contract.json').read_text());G.update(m=m,d=mujoco.MjData(m),qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]],table=m.body('demo_table').id)
def one(row):
 m,d=G['m'],G['d'];s=row.get('scene_metadata');m.body_pos[G['table']]=[*s['table_center'][:2],0.] if s else [100.,100.,0.];yaw=s['table_yaw'] if s else 0.;m.body_quat[G['table']]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];z=np.load(row['motion_path']);distance=[]
 for i in range(len(z['joint_pos'])):
  d.qpos[:7]=np.r_[z['body_pos_w'][i,0],z['body_quat_w'][i,0]];d.qpos[G['qa']]=z['joint_pos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);minimum=min((float(d.contact[j].dist) for j in range(d.ncon)),default=0.)
  if i<3:
   mujoco.mj_forward(m,d);full=min((float(d.contact[j].dist) for j in range(d.ncon)),default=0.);assert abs(full-minimum)<1e-8,(full,minimum)
  distance.append(minimum)
 v=np.asarray(distance);valid=np.flatnonzero(v[:-1]>=-.01);return dict(task=row['task'],motion_path=row['motion_path'],frames=len(v),valid_frames=len(valid),valid_start=bool(v[0]>=-.01),min_contact_distance_m=float(v.min()),penetration_over_1cm_fraction=float((v<-.01).mean())),valid
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='joint_corpus_v1');p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=D/(a.dataset+'_reset_geometry');out.mkdir(exist_ok=False);rows=json.loads((D/a.dataset/'manifest.json').read_text());reports=[];valids=[]
 with ProcessPoolExecutor(a.workers,initializer=initialize) as pool:
  for i,(report,valid) in enumerate(pool.map(one,rows)):
   reports.append(report);valids.append(valid);print(i,report['task'],report['valid_frames'],report['frames'],flush=True)
 ends=np.cumsum([len(v) for v in valids]);np.savez_compressed(out/'valid_phases.npz',valid_offsets=np.concatenate(valids),ends=ends)
 summary={}
 for task in sorted({r['task'] for r in reports}):
  rr=[r for r in reports if r['task']==task];summary[task]=dict(clips=len(rr),fully_invalid_clips=sum(r['valid_frames']==0 for r in rr),valid_start_fraction=float(np.mean([r['valid_start'] for r in rr])),mean_invalid_phase_fraction=float(np.mean([r['penetration_over_1cm_fraction'] for r in rr])))
 (out/'audit.json').write_text(json.dumps(dict(scope='Minimum signed contact geometry in the nominal simulator model. Reject phases penetrating more than1cm, including table contact preload. No trajectory edit; no proof of dynamic feasibility. First3 frames of every clip verify fast collision computation against full mj_forward.',dataset=a.dataset,results=summary,clips=reports),indent=2));print(json.dumps(summary,indent=2))
