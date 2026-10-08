"""Make the requested heading explicit while retaining source local human body pose."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
import full,smpl_direct as sm,g1_runtime as rt

def main():
    m=rt.load_model();p=sm.Policy();rows=[]
    for src in json.loads((full.OUT/'sources.json').read_text()):
        case=src['case']
        if not case.startswith(('03','05','06')):continue
        req=src['requested'];joints,quat=sm.prepare(case);q,_,root=full.native(m,case)
        t=np.maximum(np.arange(len(q))*.02-1,0)
        progress=np.clip((t-req['turn_start'])/max(req['turn_end']-req['turn_start'],.05),0,1)
        e=Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz');e[:,2]=np.deg2rad(req['turn'])*progress
        quat=Rotation.from_euler('xyz',e).as_quat()[:,[3,0,1,2]];p.set_motion(joints,quat)
        for seed in [None,5511,5512]:
            label=f'heading_{case}_s{seed}'
            r=full.score(m,p,case,label,(q,quat,root),seed,dict(requested_turn_deg=req['turn'],mode=2,local_body_pose_unchanged=True))
            r['turn_success']=bool(r['complete'] and abs(r['turn_deg']-req['turn'])<=5)
            if req['stop'] is not None:
                raw=np.load(full.OUT/'rollouts'/(label+'.npz'));start=int((1+req['stop']+.5)*50)
                r['post_stop_plus_half_second_mean_speed_mps']=float(np.linalg.norm(raw['qvel'][start:,:2],axis=1).mean())
            rows.append(r);full.save('heading_results.json',rows)
if __name__=='__main__':main()
