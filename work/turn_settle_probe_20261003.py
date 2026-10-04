"""Development diagnostic only: hold final pose 2 s; retain original task-window metric."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work');import confirmation_eval_20261003 as ce
import numpy as np
ce.init();tr=ce.g.tr;m=ce.g.M;out=ce.ROOT/'turn_settle_development';out.mkdir(exist_ok=True);rows=[r for r in json.loads((ce.ROOT/'physical_development_manifest.json').read_text()) if r['task']=='turn'];results=[]
def heading(states):
 p=tr.get_positions(m,states);v=p[:,1,:2]-p[:,2,:2];return np.unwrap(np.arctan2(v[:,1],v[:,0]))
wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
for row in rows:
 ref=ce.g.convert(np.load(row['path']),'uniform');long=np.r_[ref,np.repeat(ref[-1:],40,axis=0)];arrays,fall=tr.prior_run.rollout(m,ce.g.P,*tr.upsample(long[:,7:],long[:,3:7],long[:,:3]));ts=(np.arange(len(arrays['qpos']))+1)*.02;act=heading(arrays['qpos']);target=heading(ref);start=np.interp(1,ts,act);r=dict(row,fall=fall,reference_Q=float(-(target[-1]-target[0])),actual_turn_after_extra_s={str(extra):float(-(np.interp(6.95+extra,ts,act)-start)) for extra in [0,1,2]},absolute_final_heading_error_after_extra_s={str(extra):float(wrap(np.interp(6.95+extra,ts,act)-target[-1])) for extra in [0,1,2]});results.append(r);np.savez_compressed(out/(Path(row['path']).stem+'.npz'),time_s=ts,reference_qpos=long,**arrays);print(r,flush=True)
(out/'results.json').write_text(json.dumps(dict(scope='Development diagnostic; extra hold is NOT a changed task scoring window and not confirmation evidence',results=results),indent=2))
