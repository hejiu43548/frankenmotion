import os
os.environ['MUJOCO_GL']='egl'
import json,hashlib
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/universal_tracker_20261006';O=R/'outputs_amass/visual_audit_20261007';O.mkdir(exist_ok=True)
rows=json.loads((U/'general_evaluation/fresh_candidate/audited_results.json').read_text());selected=[r for r in rows if r['source']==r['task']+'_p0_s0' and r['command_index'] in [0,2,4]]
parents=np.load(R/'outputs_amass/franken_eleven_20261003/skeleton.npz')['parents'][:22].tolist()+[20,21]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
xml='<mujoco><visual><global offwidth="480" offheight="400"/></visual><worldbody><light pos="0 0 4"/><geom type="plane" size="20 20 .1" rgba=".22 .25 .28 1"/></worldbody></mujoco>'
hm=mujoco.MjModel.from_xml_string(xml);hd=mujoco.MjData(hm);mujoco.mj_forward(hm,hd)
units=dict(zip(['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'],['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s']));tol=dict(zip(units,[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05]));records=[]
for task in units:
 rr=[r for r in selected if r['task']==task];keys=[]
 with imageio.get_writer(str(O/(task+'.mp4')),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
  for row in rr:
   run=Path(row['run']);a=np.load(run/'actual.npz')['qpos'];ref=np.load(run/'motion.npz');h=np.load(row['path'])['joints_zup_m'];c=json.loads((run/'inference_contract.json').read_text());m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=480;m.vis.global_.offheight=400
   for j in range(m.ngeom):
    if m.geom(j).name=='target_mark':m.geom_rgba[j,3]=0
   qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]]
   cams=[]
   for points in [h[:,0],ref['body_pos_w'][:,0],a[:,:3]]:
    cam=mujoco.MjvCamera();cam.azimuth=125;cam.elevation=-15;cam.lookat[:]=[(points[:,0].min()+points[:,0].max())/2,(points[:,1].min()+points[:,1].max())/2,.8];cam.distance=max(3.,np.ptp(points[:,:2],axis=0).max()+1.8);cams.append(cam)
   cams[2].lookat[:]=cams[1].lookat;cams[2].distance=cams[1].distance
   indices=list(range(0,len(ref['joint_pos']),2));keyidx=set(np.linspace(0,len(indices)-1,6,dtype=int).tolist())
   with mujoco.Renderer(hm,height=400,width=480) as hr,mujoco.Renderer(m,height=400,width=480) as mr:
    for ordinal,i in enumerate(indices):
     hf=np.clip((i/50-1)*20,0,len(h)-1);lo=int(hf);hi=min(lo+1,len(h)-1);pos=h[lo]*(1-(hf-lo))+h[hi]*(hf-lo)
     hr.update_scene(hd,camera=cams[0]);sc=hr.scene
     for j,p in enumerate(parents):
      if j==0:continue
      g=sc.geoms[sc.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).reshape(-1),np.array([.1,.7,.9,1.]));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.018,pos[int(p)],pos[j]);sc.ngeom+=1
     canvas=Image.new('RGB',(1440,492),'#f5f7fa');canvas.paste(Image.fromarray(hr.render()),(0,92))
     for col in [1,2]:
      if col==1:d.qpos[:7]=np.r_[ref['body_pos_w'][i,0],ref['body_quat_w'][i,0]];d.qpos[qa]=ref['joint_pos'][i]
      else:d.qpos[:]=a[min(i,len(a)-1)]
      mujoco.mj_forward(m,d);mr.update_scene(d,camera=cams[col]);canvas.paste(Image.fromarray(mr.render()),(480*col,92))
     draw=ImageDraw.Draw(canvas);draw.text((12,6),f'{task} | command {row["command"]:.3f} {units[task]} | t={i/50:.2f}s | 1x real time | first source; no success filtering',font=font,fill='#142a3b')
     vals=[row['human'],row['g1'],row['actual']]
     for col,label in enumerate(['FrankenMotion 24-joint human reference','GMR G1 reference / kinematic','Frozen shared tracker / physical rollout']):
      draw.text((480*col+10,33),label,font=small,fill='#142a3b');v=vals[col];q=v['quantity'] if v else float('nan');passed=bool(v and v['event_pass'] and abs(q-row['command'])<=tol[task]);draw.text((480*col+10,57),f'Q={q:.3f} | numeric+event: {"PASS" if passed else "MISS"}',font=small,fill='#087258' if passed else '#b23434')
     if i<50:draw.text((10,76),'Human held during robot initialization',font=small,fill='#526373')
     if i>=len(a):draw.text((970,76),'TERMINATED: final state held',font=small,fill='red')
     writer.append_data(np.asarray(canvas))
     if ordinal in keyidx:keys.append(canvas.resize((960,328)))
   records.append(dict(task=task,command=row['command'],human=row['human'],g1=row['g1'],actual=row['actual'],run=str(run),human_path=row['path'],raw_sha256=hashlib.sha256((run/'actual.npz').read_bytes()).hexdigest()))
 sheet=Image.new('RGB',(1920,328*((len(keys)+1)//2)),'white')
 for k,im in enumerate(keys):sheet.paste(im,((k%2)*960,(k//2)*328))
 sheet.save(O/(task+'_inspection.jpg'));print(task,flush=True)
(O/'protocol.json').write_text(json.dumps(dict(selection='Every task: first frozen fresh-test prompt/noise p0_s0; commands c0,c2,c4. No numeric-success filtering.',tracker='Original frozen unified actor; jump candidate not substituted',fps=25,playback_speed=1,human='24 joints from generated joints_zup_m; original skeleton parent hierarchy plus hand endpoints, not PHC controller simulation',records=records),indent=2))
