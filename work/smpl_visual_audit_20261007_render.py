import os
os.environ['PYOPENGL_PLATFORM']='egl';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib, numpy as np,torch,smplx,pyrender,trimesh,imageio.v2 as imageio
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');V=R/'outputs_amass/visual_audit_20261007';O=R/'outputs_amass/smpl_visual_audit_20261007';O.mkdir(exist_ok=True)
torch.set_num_threads(1);weight=R/'deps/smplh/SMPLH_MALE.npz';m=smplx.SMPLH(str(weight),ext='npz',use_pca=False,flat_hand_mean=True,num_betas=10).eval();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
rows=json.loads((R/'outputs_amass/universal_tracker_20261006/general_evaluation/fresh_candidate/audited_results.json').read_text());rows=[r for r in rows if r['source']==r['task']+'_p0_s0' and r['command_index'] in [0,2,4]]
renderer=pyrender.OffscreenRenderer(480,400);scene=pyrender.Scene(bg_color=[.90,.93,.96,1],ambient_light=[.45,.45,.45]);floor=trimesh.creation.box(extents=[30,30,.015]);floor.apply_translation([0,0,-.015]);scene.add(pyrender.Mesh.from_trimesh(floor,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.6,.65,.7,1],roughnessFactor=1)))
cam=pyrender.PerspectiveCamera(yfov=np.pi/4);cn=scene.add(cam);light=pyrender.DirectionalLight(color=np.ones(3),intensity=3);ln=scene.add(light);body=None;records=[]
def camera(target,distance=3):
 az=np.radians(125);el=np.radians(15);eye=np.array(target)+distance*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)]);back=eye-target;back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right);mat=np.eye(4);mat[:3,:3]=np.stack([right,up,back],1);mat[:3,3]=eye;return mat
for task in dict.fromkeys(r['task'] for r in rows):
 reader=imageio.get_reader(str(V/(task+'.mp4')));olditer=iter(reader);keys=[];count=0
 with imageio.get_writer(str(O/(task+'.mp4')),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
  for row in [r for r in rows if r['task']==task]:
   z=np.load(row['path']);pose=torch.from_numpy(z['poses_axisangle']).float();n=len(pose)
   with torch.no_grad():out=m(global_orient=pose[:,:3],body_pose=pose[:,3:],left_hand_pose=torch.zeros(n,45),right_hand_pose=torch.zeros(n,45),transl=torch.tensor(z['root_translation']).float(),betas=torch.zeros(n,10))
   vertices=out.vertices.numpy();error=float(np.max(abs(out.joints.numpy()[:,:22]-z['joints_zup_m'][:,:22])));assert error<1e-3,error
   ref=np.load(Path(row['run'])/'motion.npz');indices=list(range(0,len(ref['joint_pos']),2));samples=set(np.linspace(0,len(indices)-1,6,dtype=int));h=z['joints_zup_m'];target=np.array([(h[:,0,0].min()+h[:,0,0].max())/2,(h[:,0,1].min()+h[:,0,1].max())/2,.8]);target[2]=float((vertices[:,:,2].min()+vertices[:,:,2].max())/2);distance=max(3.6,np.ptp(h[:,0,:2],axis=0).max()+1.8)
   for ordinal,i in enumerate(indices):
    hf=np.clip((i/50-1)*20,0,n-1);lo=int(hf);hi=min(lo+1,n-1);alpha=hf-lo;verts=vertices[lo]*(1-alpha)+vertices[hi]*alpha
    if body is not None:scene.remove_node(body)
    body=scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=verts,faces=m.faces,process=False),material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.15,.48,.7,1],roughnessFactor=.7),smooth=True))
    if task in ['walk','back_walk','sidestep']:target[:2]=h[lo,0,:2]*(1-alpha)+h[hi,0,:2]*alpha;distance=3.6
    mat=camera(target,distance);scene.set_pose(cn,mat);scene.set_pose(ln,mat);rgb,_=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    old=next(olditer);canvas=Image.fromarray(old);canvas.paste(Image.fromarray(rgb[:,:,:3]),(0,92));draw=ImageDraw.Draw(canvas);draw.rectangle((0,28,479,53),fill='#f5f7fa');draw.text((10,32),'FrankenMotion SMPL-H mesh / kinematic',font=font,fill='#142a3b');writer.append_data(np.asarray(canvas));count+=1
    if ordinal in samples:keys.append(canvas.resize((960,328)))
   records.append(dict(task=task,command=row['command'],path=row['path'],max_joint_fk_error_m=error,frames=n,run=row['run']))
 reader.close();sheet=Image.new('RGB',(1920,328*((len(keys)+1)//2)),'white')
 for k,im in enumerate(keys):sheet.paste(im,((k%2)*960,(k//2)*328))
 sheet.save(O/(task+'_inspection.jpg'));print(task,count,flush=True)
renderer.delete();(O/'render_protocol.json').write_text(json.dumps(dict(model=str(weight),model_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),model_type='SMPL-H male, zero shape, flat neutral hands; finger motion not generated',pose='Exact saved axis-angle and root translation, no fitting, smoothing or grounding. 20Hz source mesh linearly interpolated to25fps display, same as prior skeleton display timing. First1s held for robot initialization.',selection='Same first source, c0/c2/c4 as prior video; no failure filtering',records=records),indent=2))
