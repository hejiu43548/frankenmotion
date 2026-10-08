from uc_common import *
from unified_control import SharedCommands,load_base
from legacy_targets import LegacyTargets
from sampling import generate
import argparse,hashlib
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--teachers',action='store_true');p.add_argument('--checkpoint');p.add_argument('--full',action='store_true');p.add_argument('--layouts',type=int,default=4);p.add_argument('--seed',type=int,default=107086000);a=p.parse_args();assert a.teachers!=bool(a.checkpoint);out=D/'table'/a.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(3)
if a.teachers:controller=LegacyTargets();h='frozen_baseline'
elif a.full:
 from single_runtime import load_one_checkpoint
 m=load_one_checkpoint(a.checkpoint);h=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
else:ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);controller=SharedCommands(ck['width']);controller.load_state_dict(ck['controller']);h=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
if not a.full:m=load_base(controller)
fk=FK();rng=np.random.default_rng(a.seed);layouts=[(float(rng.uniform(.85,1.75)),float(rng.uniform(-.5,.5))) for _ in range(a.layouts)]
for folder in ['walks','away','reaches','turns']:(out/folder).mkdir()
def save(raw,path,task,cmd):
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.,task=task,command=cmd)
for folder,commands,offset in [('walks',layouts,0),('away',[(1.2,0.)]*a.layouts,3000)]:
 rows=[]
 for i,(distance,angle) in enumerate(commands):
  hd=distance*fk.height/1.0486437524221748;goal=[hd,angle];target=hd*np.array([np.cos(angle),np.sin(angle)]);hist=[];best=None;best_error=float('inf')
  for iteration in range(4):
   raw,_,_=generate(m,'walk_endpoint',[goal[0]],0,[a.seed+i+offset],[goal]);p=fk(raw)[0].numpy();xy=p[-1,0,:2]-p[0,0,:2];err=xy-target;norm=float(np.linalg.norm(err));hist.append(dict(injected=goal,actual_xy=xy.tolist(),error=norm))
   if norm<best_error:best=raw.clone();best_error=norm;best_iteration=iteration
   if norm<.015:break
   updated=goal[0]*np.array([np.cos(goal[1]),np.sin(goal[1])])-err;goal=[float(np.clip(np.linalg.norm(updated),.8,2.8)),float(np.clip(np.arctan2(updated[1],updated[0]),-.65,.65))]
  dest=out/folder/f'scene_{i:03d}';dest.mkdir();save(best,dest/'human_walk.npz','walk',distance/4.3);f=np.array([np.cos(angle),np.sin(angle)]);right=np.array([np.sin(angle),-np.cos(angle)]);xy=distance*f;center=xy+.65*f;target=xy+.4*f+.14*right;row=dict(index=i,seed=a.seed+i+offset,distance_robot_m=distance,direction_rad=angle,distance_human_m=hd,goal_xy=xy.tolist(),table_center=[*center,.4],table_half_size=[.4,.48,.035],table_top=.8,table_yaw=angle,hand_target=[*target,.84],source=str(dest),weight_sha256=h,generation_refinements=hist,selected_generation_iteration=best_iteration);(dest/'scene.json').write_text(json.dumps(row,indent=2));rows.append(row);print(folder,i,best_error,flush=True)
 (out/folder/'manifest.json').write_text(json.dumps(rows,indent=2))
rows=[]
for i in range(a.layouts):
 for j,c in enumerate([.3,.5]):
  dest=out/'reaches'/f'scene_{2*i+j:03d}';dest.mkdir();raw,_,_=generate(m,'place_hold_retract',[c],0,[a.seed+1000+i],[.84]);save(raw,dest/'human_reach.npz','reach',c);rows.append(dict(index=2*i+j,source=str(dest),command=c,exit_task='none',exit_command=0.,seed=a.seed+1000+i,walk_source=str(out/'walks'/f'scene_{i:03d}'),generator_sha256=h))
 desired=math.radians([90,135][i%2]);injected=desired;hist=[]
 for iteration in range(4):
  raw,_,_=generate(m,'turn_endpoint',[injected],0,[a.seed+2000+i]);p=fk(raw)[0].numpy();side=p[:,1,:2]-p[:,2,:2];yy=np.unwrap(np.arctan2(side[:,1],side[:,0]));actual=float(-(yy[-1]-yy[0]));hist.append(dict(injected=injected,actual=actual))
  if iteration<3:injected=float(np.clip(injected+desired-actual,0,4.5))
 save(raw,out/'turns'/f'turn_{i:03d}.npz','turn',desired);(out/'turns'/f'turn_{i:03d}.json').write_text(json.dumps(hist,indent=2));print('reach_turn',i,flush=True)
(out/'reaches'/'manifest.json').write_text(json.dumps(rows,indent=2));(out/'protocol.json').write_text(json.dumps(dict(seed=a.seed,layouts=layouts,checkpoint=a.checkpoint,sha256=h,steps=50,scope='Legacy full table pipeline preserved: up to3 command-input refinements based on generated endpoint/angle only, GMR, retiming, rigid placement, blends and causal root-reference anchoring. All generator segments use one fixed shared controller for student. No physical outcomes used to refine commands.',tracker='same backed-up495-input actor for both generator variants'),indent=2))
