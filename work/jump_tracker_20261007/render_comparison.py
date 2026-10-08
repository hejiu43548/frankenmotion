"""Recorded physical rollouts versus kinematic reference; fixed predeclared source."""
import os
os.environ['MUJOCO_GL']='egl'
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);p.add_argument('--source',default='jump_p0_s0');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
old=json.load(open(Path(a.baseline)/'audited_results.json'));new=json.load(open(Path(a.candidate)/'audited_results.json'))
def key(r):return (r['source'],round(float(r['command']),6))
ob={key(r):r for r in old};nb={key(r):r for r in new};selected=[nb[(a.source,v)] for v in [.25,.4,.55]]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
writer=imageio.get_writer(str(out/'fixed_source_comparison.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1);records=[]
for row in selected:
 br=ob[key(row)];cr=row;bp=Path(br['run']);cp=Path(cr['run']);ref=np.load(cp/'motion.npz');b=np.load(bp/'actual.npz')['qpos'];c=np.load(cp/'actual.npz')['qpos'];contract=json.load(open(cp/'inference_contract.json'));m=mujoco.MjModel.from_binary_path(str(cp/'scene.mjb'));d=mujoco.MjData(m)
 for gid in range(m.ngeom):
  if m.geom(gid).name=='target_mark':
   assert not m.geom_contype[gid] and not m.geom_conaffinity[gid];m.geom_rgba[gid,3]=0
 m.vis.global_.offwidth=640;m.vis.global_.offheight=480;renderer=mujoco.Renderer(m,height=480,width=640);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[0,0,.9];camera.distance=3.3;camera.azimuth=125;camera.elevation=-18
 qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in contract['joint_names']]];n=len(ref['joint_pos']);indices=list(range(0,n,2));apex=int(np.argmax(c[:,2]));crouch=int(np.argmin(c[:max(1,apex),2]));keyframes=sorted(set([indices[0],2*(crouch//2),2*(apex//2),min(indices[-1],2*((apex+15)//2)),min(indices[-1],2*((apex+30)//2)),indices[-1]]));frames=[]
 for i in indices:
  canvas=Image.new('RGB',(1920,584),'#f5f7fa');draw=ImageDraw.Draw(canvas);draw.text((18,9),f'Fresh held-out noise | command {row["command"]:.2f} m (human-equivalent) | t={i/50:.2f} s | 1x real time',font=font,fill='#152b3b')
  for col,(label,trajectory,result) in enumerate([('GMR reference / kinematic',None,None),('Original shared tracker',b,br),('New shared residual tracker',c,cr)]):
   if trajectory is None:d.qpos[:7]=np.r_[ref['body_pos_w'][i,0],ref['body_quat_w'][i,0]];d.qpos[qa]=ref['joint_pos'][i]
   else:d.qpos[:]=trajectory[min(i,len(trajectory)-1)]
   mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera);canvas.paste(Image.fromarray(renderer.render().copy()),(col*640,104));draw.text((col*640+16,43),label,font=font,fill='#152b3b')
   quantity=row['g1']['quantity'] if result is None else (result['actual']['quantity'] if result.get('actual') else None)
   status='reference only' if result is None else ('complete' if result['physical_complete'] else 'FAILED / stopped')
   if result is not None:
    success=bool(result.get('actual') and result['actual']['event_pass'] and abs(result['actual']['quantity']-row['command'])<=.04);status+=' | target '+('PASS' if success else 'MISS')
   qtext=f'{quantity:.3f} m' if quantity is not None else 'unmeasurable';draw.text((col*640+16,76),f'Height: {qtext} | {status}',font=small,fill='#b32424' if result is not None and not result['physical_complete'] else '#354c60')
   if trajectory is not None and i>=len(trajectory):draw.rectangle((col*640+4,108,col*640+635,579),outline='#d52424',width=6);draw.text((col*640+18,550),'TERMINATED — final frame held',font=small,fill='#d52424')
  writer.append_data(np.asarray(canvas))
  if i in keyframes:
   frame=out/f'command_{row["command"]:.2f}_frame_{i:04d}.png';canvas.save(frame);frames.append(canvas.resize((1440,438)))
 renderer.close();montage=Image.new('RGB',(1440,438*len(frames)),'white')
 for j,frame in enumerate(frames):montage.paste(frame,(0,j*438))
 montage.save(out/f'command_{row["command"]:.2f}_inspection.png');records.append(dict(command=row['command'],source=row['source'],baseline_run=str(bp),candidate_run=str(cp),baseline_complete=br['physical_complete'],candidate_complete=cr['physical_complete'],baseline_quantity=br.get('actual'),candidate_quantity=cr.get('actual'),candidate_raw_sha256=hashlib.sha256((cp/'actual.npz').read_bytes()).hexdigest(),keyframes=keyframes))
writer.close();(out/'render_protocol.json').write_text(json.dumps(dict(selection='Predeclared first prompt/seed, commands0.25/0.40/0.55, irrespective of outcome',source=a.source,playback_speed=1.,fps=25,reference='Kinematic GMR replay only',actual='Recorded MuJoCo physical qpos; no state edits or replacement',failure_display='Hold final failed frame with red border; retain full reference duration',records=records),indent=2))
