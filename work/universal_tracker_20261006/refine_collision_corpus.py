"""Uniform root/leg-invariant arm self-collision retargeting ablation."""
import argparse,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collision_refinement import CollisionRefiner
from fk_conversion import convert,order,model,jq,D
G={}
def initialize():
 p=D/'table_evaluation/baseline_stable/scene_000/reference_input';G['r']=CollisionRefiner(p/'scene.mjb',p/'inference_contract.json')
def run(item):
 row,out,table=item;r=G['r'];out=Path(out);dest=out/(f'scene_{row["index"]:03d}' if table else Path(row['path']).stem);dest.mkdir()
 source=Path(row['source']) if table else None
 ref_path=source/'reference_contact.npz' if table else Path(row['reference_path']);z=dict(np.load(ref_path));ref=z['reference_qpos'];native=np.repeat(model.qpos0[None],len(ref),axis=0);native[:,:7]=ref[:,:7];native[:,jq]=ref[:,7:][:,order];states,logs=r.refine(native);corrected=ref.copy();corrected[:,7:][:,order]=states[:,jq]
 fixed=[i for i in range(ref.shape[1]) if i<7 or order.index(i-7) not in [list(jq).index(a) for a in r.arm_qa]];assert np.array_equal(ref[:,fixed],corrected[:,fixed]);z['reference_qpos']=corrected
 rp=dest/'reference_contact.npz' if table else dest/'reference.npz';np.savez_compressed(rp,**z)
 (dest/'collision_audit.json').write_text(json.dumps(logs));summary=dict(frames=len(ref),before_penetration_fraction=float(np.mean([x['before_distance_m']<-.001 for x in logs])),after_penetration_fraction=float(np.mean([x['after_distance_m']<-.001 for x in logs])),max_arm_change_rad=max(x['max_arm_change_rad'] for x in logs),solver_error_frames=sum(bool(x['solver_errors']) for x in logs),fixed_coordinates_bit_exact=True)
 if table:
  inp=dest/'reference_input';shutil.copytree(source/'reference_input',inp);shutil.copy2(source/'reference_contact.json',dest/'reference_contact.json');np.savez_compressed(inp/'motion.npz',fps=50.,**convert(corrected,entry=False));result=json.loads((inp/'result.json').read_text());result['scene']['source']=str(dest);(inp/'result.json').write_text(json.dumps(result,indent=2));new=dict(row,source=str(dest),self_collision_refinement=summary)
 else:
  mp=dest/'motion.npz';np.savez_compressed(mp,fps=50.,**convert(corrected));new=dict(row,reference_path=str(rp),motion_path=str(mp),self_collision_refinement=summary)
 return new
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--table',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);rows=json.loads(Path(a.manifest).read_text());results=[]
 with ProcessPoolExecutor(a.workers,initializer=initialize) as pool:
  for row in pool.map(run,[(r,str(out),a.table) for r in rows]):
   results.append(row);(out/'manifest.json').write_text(json.dumps(results,indent=2));print(len(results),row.get('task',row.get('index')),row['self_collision_refinement'],flush=True)
