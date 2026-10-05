import os,json
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';S=D/'final_test/scene_000';out=D/'gait_diagnosis';out.mkdir(exist_ok=True);m=mujoco.MjModel.from_binary_path(str(S/'unified/scene.mjb'));d=mujoco.MjData(m);c=json.loads((S/'unified/inference_contract.json').read_text());ref=np.load(S/'unified/motion.npz');qa=[int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']];ids=[c['joint_names'].index(n) for n in ['left_knee_joint','right_knee_joint','left_hip_pitch_joint','right_hip_pitch_joint']];base=np.tile(np.asarray(c['initial_qpos']),(len(ref['joint_pos']),1));base[:,qa]=ref['joint_pos'];base[:,:3]=ref['body_pos_w'][:,0];base[:,3:7]=ref['body_quat_w'][:,0];seq={'GMR / planned reference':base,'Original shared tracker':np.load(S/'original_tracker/actual.npz')['qpos'],'Interaction-tuned tracker':np.load(S/'unified/actual.npz')['qpos']};records={}
cam=mujoco.MjvCamera();cam.distance=2.3;cam.azimuth=100;cam.elevation=-12;m.vis.global_.offwidth=640;m.vis.global_.offheight=480
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);sheet=Image.new('RGB',(1920,960),'#101b2a')
with mujoco.Renderer(m,height=320,width=480) as renderer:
 for row,(label,q) in enumerate(seq.items()):
  joints=q[100:300,qa];records[label]={'mean_joint_deg':dict(zip([c['joint_names'][i] for i in ids],np.degrees(joints[:,ids].mean(0)).tolist())),'std_joint_deg':dict(zip([c['joint_names'][i] for i in ids],np.degrees(joints[:,ids].std(0)).tolist()))}
  for col,frame in enumerate([100,150,200,250]):
   d.qpos[:]=q[frame];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3]+[0,0,-.08];renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,480,45),fill='#101b2a');dr.text((8,4),label,font=font,fill='white');dr.text((8,23),f'{frame*.02:.1f}s',font=font,fill='white');sheet.paste(im,(col*480,row*320))
records['actual_vs_reference_joint_rmse_deg']={name:float(np.degrees(np.sqrt(np.mean((q[100:300,qa]-base[100:300,qa])**2)))) for name,q in list(seq.items())[1:]}
sheet.save(out/'reference_and_trackers.png');(out/'metrics.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
