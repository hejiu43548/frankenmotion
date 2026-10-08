"""Record long-lookahead training states under ONE frozen stable actor.
The actor consumes its original 361 observations; 495 observations are stored.
Only the 42 pre-existing training scenes are used; no development/heldout groups.
"""
import json,subprocess,shutil,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';out=D/'long_retention_v2';out.mkdir(exist_ok=False)
source=(R/'work/turn_cpu_anchor_evaluate_20261005.py').read_text();source="import sys\nsys.path.insert(0,'/home/pku/frankenmotion/work')\n"+source;source=source.replace('torch.from_numpy(obs)[None]','torch.from_numpy(obs[:361])[None]').replace('str(Path(__file__).with_name("turn_metrics_20261005.py"))',"'/home/pku/frankenmotion/work/turn_metrics_20261005.py'");runtime=W/'collect_long_runtime.py';runtime.write_text(source)
rows=json.loads((R/'outputs_amass/turn_demo_20261005/training_full_v1/training_manifest.json').read_text())[:42];assert len(rows)==42
py=str(R/'work/mjlab_stable_env/bin/python');actor=D/'backup/stable_frozen/actor.pt';checkpoint=D/'backup/stable_frozen/policy.pt';template=D/'table_evaluation/baseline_stable/scene_000/reference_input'
def one(pair):
 i,row=pair;scene=out/f'scene_{i:03d}';scene.mkdir();inp=scene/'reference_input';inp.mkdir();ref=np.load(row['motion_path']);c=json.loads((template/'inference_contract.json').read_text());c['preview_offsets']=[5,10,20,35,50]
 model=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'));meta=row['scene_metadata']
 bid=model.body('demo_table').id;model.body_pos[bid]=[meta['table_center'][0],meta['table_center'][1],0.];q=Rotation.from_euler('z',meta['table_yaw']).as_quat();model.body_quat[bid]=q[[3,0,1,2]]
 data=mujoco.MjData(model);data.qpos[:7]=np.r_[ref['body_pos_w'][0,0],ref['body_quat_w'][0,0]];qa=model.jnt_qposadr[[model.joint('robot/'+n).id for n in c['joint_names']]];data.qpos[qa]=ref['joint_pos'][0];mujoco.mj_forward(model,data);c['initial_qpos']=data.qpos.tolist();c['initial_qvel']=np.zeros(model.nv).tolist()
 (inp/'inference_contract.json').write_text(json.dumps(c,indent=2));mujoco.mj_saveModel(model,str(inp/'scene.mjb'));shutil.copy2(row['motion_path'],inp/'motion.npz');(inp/'result.json').write_text(json.dumps(dict(scene=meta),indent=2));(scene/'reference_contact.json').write_text(json.dumps(row['reference_metadata'],indent=2))
 with (scene/'worker.log').open('w') as log:result=subprocess.run([py,str(runtime),'--run',str(inp),'--actor',str(actor),'--checkpoint',str(checkpoint),'--output',str(scene/'rollout')],stdout=log,stderr=subprocess.STDOUT)
 return dict(group=i,returncode=result.returncode,path=str(scene/'rollout/actual.npz'),result=str(scene/'rollout/result.json'))
records=[]
with ThreadPoolExecutor(4) as pool:
 for row in pool.map(one,enumerate(rows)):
  records.append(row);(out/'collection.json').write_text(json.dumps(records,indent=2));print(len(records),row['returncode'],flush=True)
obs=[];groups=[];retained=[]
for row in records:
 if row['returncode']:continue
 result=json.loads(Path(row['result']).read_text())
 if not result['physical_complete']:continue
 z=np.load(row['path']);assert z['observations'].shape[1]==495;obs.append(z['observations']);groups.extend([row['group']]*len(z['observations']));retained.append(row['group'])
assert obs;np.savez_compressed(out/'dataset.npz',observations=np.concatenate(obs).astype(np.float32),group=np.asarray(groups));(out/'protocol.json').write_text(json.dumps(dict(teacher=str(checkpoint),teacher_input=361,stored_observation_dim=495,requested_training_groups=list(range(42)),retained_training_groups=retained,observations=len(groups),scope='One frozen stable actor, complete training-scene rollouts only; no held-out state or original routed teacher label. Runtime reconstructs future reference features online, including causal anchoring.'),indent=2));print('Dataset ready',len(groups),len(retained),flush=True)
