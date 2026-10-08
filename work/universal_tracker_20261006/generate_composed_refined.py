"""Deterministic command-space refinement; generated motion is never edited.
Same inversion algorithm and shared generator for both composite prompts.
"""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import physical_adapter_20261003
from core import torch,np,load_model,FK,sample,encode_control,HH
out=D/'generated_composed_refined';out.mkdir(exist_ok=False);(out/'human').mkdir();torch.set_num_threads(4)
weight=R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt';model,_=load_model('cpu',weight);fk=FK('cpu');rows=[]
for task in ['walk_wave','walk_point']:
 prompt=D/'generated_demo_candidates/prompts'/(task+'.json');z=torch.load(prompt.with_suffix('.pt'),map_location='cpu',weights_only=False)
 for si in range(2):
  seed=96061000+si
  for ci,target in enumerate([.35,.55]):
   lo,hi=.15,1.1;injected=target;history=[];best=None
   for iteration in range(9):
    cmd=torch.tensor([injected]);values=torch.zeros(1,120,2);values[...,0]=cmd[:,None]*fk.height/HH;controls=encode_control(values,torch.ones(1,120,dtype=torch.bool))
    raw=sample(model,z['local'][None],z['tx'],10,cmd,[seed],True,root_controls=controls)
    with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
    p=pos[0,:,0];q=float(torch.linalg.vector_norm(p[1:,:2]-p[:-1,:2],dim=-1).sum()/(119/20)*HH/fk.height)
    history.append(dict(iteration=iteration,injected_command=injected,measured_human_equivalent_path_speed=q))
    if best is None or abs(q-target)<best[0]:best=(abs(q-target),raw.clone(),pos.clone(),poses.clone(),root.clone(),injected,q)
    if abs(q-target)<.015:break
    if q<target:lo=injected
    else:hi=injected
    injected=(lo+hi)/2
   error,raw,pos,poses,root,injected,q=best;path=out/'human'/f'{task}_s{si}_c{ci}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
   rows.append(dict(task=task,source=task+f'_s{si}',seed=seed,command=target,command_index=ci,path=str(path),prompt=str(prompt),requested_command=target,injected_command=injected,refinement_history=history,human_path_speed_m_s=q,command_error=error,numeric_command='human-equivalent path speed; numerical inversion in generator command space only',source_type='FrankenMotion_generated_with_command_refinement',split='development',frames=120,generator=str(weight),generator_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(task,si,target,'injected',round(injected,3),'achieved',round(q,3),flush=True)
(out/'protocol.json').write_text(json.dumps(dict(scope='Same-noise bisection over numeric generator input. All output frames are direct generator samples, no pose/root editing and no simulator feedback. Both composite tasks use one generator checkpoint. Development demo candidates, not proof of intrinsic command calibration.',max_queries=9,tolerance=.015,search_bounds=[.15,1.1],shared_weights=str(weight)),indent=2))
