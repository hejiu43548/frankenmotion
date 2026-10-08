"""Geometric feasibility diagnostic of frozen references, not physical rollout."""
import json
from pathlib import Path
import numpy as np,mujoco
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');results=[]
for scene in sorted((D/'table_evaluation/baseline_stable').glob('scene_*')):
 run=scene/'reference_input';c=json.loads((run/'inference_contract.json').read_text());m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);ref=np.load(run/'motion.npz');segments=json.loads((scene/'reference_contact.json').read_text())['segments'];qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]];hand=m.geom('robot/right_hand_collision').id;table=m.geom('table_top').id
 rows=[]
 for i in range(len(ref['joint_pos'])):
  d.qpos[:7]=np.r_[ref['body_pos_w'][i,0],ref['body_quat_w'][i,0]];d.qpos[qa]=ref['joint_pos'][i];mujoco.mj_forward(m,d)
  distances=[float(d.contact[j].dist) for j in range(d.ncon) if hand in d.contact[j].geom and table in d.contact[j].geom]
  rows.append(min(distances,default=0.))
 rows=np.array(rows);record=dict(scene=scene.name)
 for key in ['contact_hold','retract_lower']:
  lo,hi=segments[key];values=rows[int(lo*2.5):int(hi*2.5)];record[key]=dict(min_signed_contact_distance_m=float(values.min()),penetration_over_5mm_s=float((values<-.005).sum()*.02),frames=len(values))
 results.append(record)
out=dict(scope='Frozen reference qpos replay solely for collision geometry. Negative distance denotes geometric penetration; no claim about reference dynamics or rollout forces.',results=results)
(D/'reference_table_contact_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
