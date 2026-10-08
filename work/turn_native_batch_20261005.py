"""Build scenes from frozen native model and run one fixed actor, with causal command placement."""
import argparse,json,subprocess,sys,shutil,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--name',required=True);p.add_argument('--checkpoint',default=str(D/'distill_full_v1/model_6000.pt'));p.add_argument('--actor',default=str(D/'distill_full_v1/actor_6000.pt'));a=p.parse_args();folder=Path(a.folder);template=D/'development_v2/scene_000/distill_v1';rows=json.loads((folder/'manifest.json').read_text());results=[]
for row in rows:
 scene=Path(row['source']);inp=scene/'reference_input'
 if not inp.exists():
  inp.mkdir();m=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'));m.body_pos[m.body('demo_table').id]=[*row['table_center'][:2],0];m.body_quat[m.body('demo_table').id]=Rotation.from_euler('z',row['table_yaw']).as_quat()[[3,0,1,2]];m.geom_pos[m.geom('target_mark').id]=[*row['hand_target'][:2],row['table_top']+.0008];mujoco.mj_saveModel(m,str(inp/'scene.mjb'),None);ref=np.load(scene/'motion.npz');c=json.loads((template/'inference_contract.json').read_text());q=m.qpos0.copy();q[:3]=ref['body_pos_w'][0,0];q[3:7]=ref['body_quat_w'][0,0];qa=[int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']];q[qa]=ref['joint_pos'][0];c['initial_qpos']=q.tolist();c['initial_qvel']=np.zeros(m.nv).tolist();(inp/'inference_contract.json').write_text(json.dumps(c,indent=2));shutil.copy2(scene/'motion.npz',inp/'motion.npz');(inp/'result.json').write_text(json.dumps(dict(scene=row,physical_complete=False,success=False,scope='Unexecuted reference input, not a demo result.'),indent=2))
 dest=scene/a.name
 with (scene/(a.name+'.log')).open('w') as log:subprocess.run([sys.executable,str(R/'work/turn_cpu_anchor_evaluate_20261005.py'),'--run',str(inp),'--actor',a.actor,'--checkpoint',a.checkpoint,'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
 for seg in ['walk','away']:subprocess.run([sys.executable,str(R/'work/turn_jitter_20261005.py'),'--run',str(dest),'--segment',seg],stdout=subprocess.DEVNULL,check=True)
 result=json.loads((dest/'result.json').read_text());results.append(result);summary=dict(planned=len(rows),processed=len(results),successes=sum(r['success'] for r in results),physical_complete=sum(r['physical_complete'] for r in results),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),results=results);(folder/(a.name+'_summary.json')).write_text(json.dumps(summary,indent=2));print(row['index'],result['success'],{k:result['command_metrics'][k] for k in ['actual_turn_deg','actual_away_displacement_m','actual_hold_wrist_forward_human_equiv_m']},flush=True)
print('COMPLETE',summary['successes'],summary['planned'],flush=True)
