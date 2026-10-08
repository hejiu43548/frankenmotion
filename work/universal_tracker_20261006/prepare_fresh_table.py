"""Frozen reference assembly; no tracker-based sample selection or pose repair."""
import sys,json,subprocess,shutil,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from fk_conversion import convert,R,D
folder=D/'fresh_table';W=R/'work/universal_tracker_20261006';selection=D/'frozen_unified/protocol.json';assert hashlib.sha256(selection.read_bytes()).hexdigest()==json.loads((folder/'generation_protocol.json').read_text())['selection_sha256']
for path in ['walk','away']:
 subprocess.run([sys.executable,str(R/'work/reach_prepare_walk_20261005.py'),'--folder',str(folder/path)],check=True,cwd=R)
subprocess.run([sys.executable,str(R/'work/reach_prepare_corpus_v3_20261005.py'),'--folder',str(folder/'reach')],check=True,cwd=R)
src=(R/'work/turn_compose_final_20261005.py').read_text();src=src.replace("D=R/'outputs_amass/turn_demo_20261005'","D=R/'outputs_amass/universal_tracker_20261006/fresh_table'").replace("D/'final_reach'","D/'reach'").replace("final_away/","away/").replace("turn_final/","turn/").replace('93005000','107064000');script=folder/'compose_frozen.py';script.write_text(src);subprocess.run([sys.executable,str(script),'--name','sequences'],check=True,cwd=R)
rows=json.loads((folder/'sequences/manifest.json').read_text());template=D/'table_evaluation/baseline_stable/scene_000/reference_input'
for row in rows:
 path=Path(row['source']);ref=np.load(path/'reference_contact.npz')['reference_qpos'];motion=convert(ref,entry=False);inp=path/'reference_input';inp.mkdir();m=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'));m.body_pos[m.body('demo_table').id]=[*row['table_center'][:2],0.];m.body_quat[m.body('demo_table').id]=Rotation.from_euler('z',row['table_yaw']).as_quat()[[3,0,1,2]];m.geom_pos[m.geom('target_mark').id]=[*row['hand_target'][:2],row['table_top']+.0008];mujoco.mj_saveModel(m,str(inp/'scene.mjb'));c=json.loads((template/'inference_contract.json').read_text());q=m.qpos0.copy();q[:7]=np.r_[motion['body_pos_w'][0,0],motion['body_quat_w'][0,0]];qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]];q[qa]=motion['joint_pos'][0];c['initial_qpos']=q.tolist();c['initial_qvel']=np.zeros(m.nv).tolist();(inp/'inference_contract.json').write_text(json.dumps(c,indent=2));np.savez_compressed(inp/'motion.npz',fps=50.,**motion);(inp/'result.json').write_text(json.dumps(dict(scene=row,physical_complete=False,success=False,scope='Fresh unexecuted reference; not a physical result.'),indent=2));print('Prepared',row['index'],flush=True)
(folder/'prepared.json').write_text(json.dumps(dict(scenes=len(rows),scope='All12 paired fresh scenes retained. Existing GMR:uniform walking/turning, wrist80/orientation5 reach. Existing full-walk retiming, rigid clip placement, transition blends and finalnominalstance. No scene handIK, no postgeneration turn-angle edits. Tracker execution notyetperformed.',selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest()),indent=2))
