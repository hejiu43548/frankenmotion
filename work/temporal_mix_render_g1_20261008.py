"""Reference versus recorded physical replay, at real time, with visible failure labels."""
import os
os.environ['MUJOCO_GL']='egl'
import argparse,json
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--frames-only',action='store_true');p.add_argument('--follow',action='store_true');a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
r=json.loads((run/'result.json').read_text());c=json.loads((run/'inference_contract.json').read_text());actual=np.load(run/'actual.npz');ref=np.load(run/'motion.npz');model=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));data=mujoco.MjData(model)
for gid in range(model.ngeom):
 if model.geom(gid).name=='target_mark':
  assert model.geom_contype[gid]==0 and model.geom_conaffinity[gid]==0;model.geom_rgba[gid,3]=0
model.vis.global_.offwidth=640;model.vis.global_.offheight=480
qa=model.jnt_qposadr[[model.joint('robot/'+n).id for n in c['joint_names']]]
renderer=mujoco.Renderer(model,height=480,width=640);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
xy=ref['body_pos_w'][:,0,:2];center=(xy.min(0)+xy.max(0))/2;span=float(np.max(np.ptp(xy,axis=0)));camera.lookat[:]=[center[0],center[1],.7];camera.distance=max(2.5,span+2);camera.azimuth=125;camera.elevation=-18
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(font_path,20);small=ImageFont.truetype(font_path,16)
indices=list(range(0,len(actual['qpos']),2));key_indices=np.linspace(0,len(indices)-1,6,dtype=int);keys=[]
writer=None if a.frames_only else imageio.get_writer(str(out/'reference_vs_physics.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
for ordinal,index in enumerate(indices):
 if a.frames_only and ordinal not in key_indices:continue
 if a.follow:
  camera.lookat[:]=[ref['body_pos_w'][index,0,0],ref['body_pos_w'][index,0,1],.7];camera.distance=2.6
 data.qpos[:7]=np.r_[ref['body_pos_w'][index,0],ref['body_quat_w'][index,0]];data.qpos[qa]=ref['joint_pos'][index];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);left=renderer.render().copy()
 data.qpos[:]=actual['qpos'][index];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);right=renderer.render().copy()
 canvas=Image.new('RGB',(1280,560),'#f6f8fb');canvas.paste(Image.fromarray(left),(0,80));canvas.paste(Image.fromarray(right),(640,80));draw=ImageDraw.Draw(canvas)
 cmd=r.get('command');title=r['source']+' | W1-5s L2-4s R3-5s'
 draw.text((16,8),title+' | t='+f'{index/50-1:.2f}s (motion clock)'+' | 1x real time',font=font,fill='#13283c')
 draw.text((16,43),'GMR reference (kinematic replay)',font=small,fill='#28465d');status='completed' if r['physical_complete'] else 'terminated at '+f'{r["termination_time"]:.2f}s'
 draw.text((656,43),'G1 physical rollout | '+status,font=small,fill='#28465d')
 if writer:writer.append_data(np.asarray(canvas))
 if ordinal in key_indices:
  canvas.save(out/f'frame_{index:04d}.png');keys.append(canvas.resize((640,280)))
if writer:writer.close()
renderer.close();montage=Image.new('RGB',(1280,280*((len(keys)+1)//2)),'white')
for i,im in enumerate(keys):montage.paste(im,((i%2)*640,(i//2)*280))
montage.save(out/'inspection_montage.png');(out/'render_protocol.json').write_text(json.dumps(dict(source_run=str(run),source_result=r,playback_speed=1.,fps=25,hidden_noncolliding_legacy_marker='target_mark',camera_mode='reference-follow' if a.follow else 'fixed-world',reference_panel='kinematic replay, not simulation',actual_panel='saved physical qpos, never replaced with reference',frames=len(indices),camera=dict(distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation,lookat=camera.lookat.tolist())),indent=2));print(out,flush=True)
