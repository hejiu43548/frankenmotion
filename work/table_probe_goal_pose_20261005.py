import sys,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'));import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample
s=json.loads((D/'final_test/scene_000/scene.json').read_text());src=next(x for x in json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text()) if x['source']=='walk_p0_s0');c=s['generation_refinements'][s['selected_generation_iteration']];goals=[[c['injected_distance_human_m'],c['injected_direction_rad']]];m=ga.load(D/'frozen/goal_adapter.pt');fk=FK('cuda');torch.set_num_threads(2);local,tx,cmd,controls,g=ga.setup(src,goals);m.denoiser.goal=None;raw=sample(m,local,tx,10,cmd,[s['seed']],True,root_controls=controls)
with torch.no_grad():p=fk(raw,canonical=False).cpu().numpy()[0]
new=np.load(D/'final_test/scene_000/human_walk.npz')['joints_zup_m'];out=D/'gait_diagnosis';np.savez_compressed(out/'goal_off_human.npz',joints_zup_m=p);report={}
for name,pts in [('goal_off_same_conditions',p),('goal_on_saved',new)]:
 a=pts[:,[1,2]]-pts[:,[4,5]];b=pts[:,[7,8]]-pts[:,[4,5]];flex=180-np.degrees(np.arccos(np.clip((a*b).sum(-1)/np.linalg.norm(a,axis=-1)/np.linalg.norm(b,axis=-1),-1,1)));feet=pts[:,[7,8]];v=np.diff(feet,axis=0)*20;low=feet[1:,:,2].argmin(-1);slip=np.linalg.norm(v[np.arange(len(v)),low,:2],axis=-1)
 report[name]=dict(knee_mean_degrees=flex[20:100].mean(0).tolist(),knee_std_degrees=flex[20:100].std(0).tolist(),lower_ankle_horizontal_speed_proxy_m_s=float(slip[20:100].mean()),endpoint_xy=(pts[-1,0,:2]-pts[0,0,:2]).tolist())
(out/'generator_pose_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
