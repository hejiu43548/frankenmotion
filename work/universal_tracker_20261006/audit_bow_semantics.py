"""Descriptive torso-angle audit; no post-hoc semantic success threshold."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')];import transfer as tr
from audit_results import canonical
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();folder=Path(a.run);source=tr.rt.load_model();records=[]
def describe(pos):
 pos=canonical(pos);torso=(pos[:,16]+pos[:,17])/2-pos[:,0];angle=np.degrees(np.arctan2(torso[:,0],torso[:,2]));return dict(initial_forward_torso_deg=float(angle[0]),max_forward_torso_deg=float(angle.max()),forward_increase_deg=float(angle.max()-angle[0]),final_forward_torso_deg=float(angle[-1]))
for row in json.loads((folder/'results.json').read_text()):
 run=Path(row['run']);human=np.load(row['path'])['joints_zup_m'];ref=np.load(row['reference_path'])['reference_qpos'];native=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));actual=np.load(run/'actual.npz')['qpos'];order=[native.jnt_qposadr[native.joint('robot/'+source.joint(i).name).id] for i in range(1,source.njnt)];states=np.c_[actual[:,:7],actual[:,order]];pos=tr.get_positions(source,states);times=np.arange(len(pos))*.02;want=1+np.arange(len(ref))*.05;observed=want<=times[-1]+1e-8;aligned=np.stack([np.interp(want[observed],times,v) for v in pos.reshape(len(pos),-1).T],1).reshape(-1,24,3);records.append(dict(source=row['source'],physical_complete=row['physical_complete'],human=describe(human),g1=describe(tr.get_positions(source,ref)),actual=describe(aligned)))
(folder/'bow_semantics.json').write_text(json.dumps(dict(scope=__doc__,definition='Forward angle of pelvis-to-midshoulder vector in each sequence initial canonical frame. Descriptor only, not a numeric bow command or a semantic success certificate.',results=records),indent=2));print(json.dumps(records,indent=2))
