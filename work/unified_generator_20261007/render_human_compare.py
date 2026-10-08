import os
os.environ['PYOPENGL_PLATFORM']='egl';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,numpy as np,torch,smplx,pyrender,trimesh,imageio.v2 as imageio
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--baseline',default='teacher_development');p.add_argument('--tasks',nargs='+',default=['jump','kick','walk']);a=p.parse_args();out=D/'visuals'/a.name;out.mkdir(parents=True,exist_ok=False)
base=json.loads((D/'generation'/a.baseline/'audited_results.json').read_text());new=json.loads((D/'generation'/a.name/'audited_results.json').read_text());key=lambda r:(r['task'],r['source'],r['command_index']);lookup={key(r):r for r in base};selected=[r for r in new if r['task'] in a.tasks and r['source']==r['task']+'_p0_s0' and r['command_index'] in [0,2,4]]
torch.set_num_threads(1);model=smplx.SMPLH(str(R/'deps/smplh/SMPLH_MALE.npz'),ext='npz',use_pca=False,flat_hand_mean=True,num_betas=10).eval();renderer=pyrender.OffscreenRenderer(480,400);scene=pyrender.Scene(bg_color=[.92,.94,.96,1],ambient_light=[.5,.5,.5]);floor=trimesh.creation.box(extents=[30,30,.015]);floor.apply_translation([0,0,-.015]);scene.add(pyrender.Mesh.from_trimesh(floor,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.58,.63,.68,1],roughnessFactor=1)));cn=scene.add(pyrender.PerspectiveCamera(yfov=np.pi/4));ln=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=3));body=None;F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);records=[]
def cam(target):
 az=np.radians(125);el=np.radians(15);eye=np.array(target)+3.6*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)]);back=eye-target;back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right);m=np.eye(4);m[:3,:3]=np.stack([right,up,back],1);m[:3,3]=eye;return m
for task in a.tasks:
 rr=[r for r in selected if r['task']==task];frames=[]
 with imageio.get_writer(str(out/(task+'.mp4')),fps=20,codec='libx264',quality=8,macro_block_size=1) as writer:
  for r in rr:
   pair=[lookup[key(r)],r];meshes=[];joints=[]
   for row in pair:
    z=np.load(row['path']);pose=torch.tensor(z['poses_axisangle']).float();n=len(pose)
    with torch.no_grad():o=model(global_orient=pose[:,:3],body_pose=pose[:,3:],transl=torch.tensor(z['root_translation']).float(),betas=torch.zeros(n,10),left_hand_pose=torch.zeros(n,45),right_hand_pose=torch.zeros(n,45))
    assert np.max(abs(o.joints.numpy()[:,:22]-z['joints_zup_m'][:,:22]))<.001
    meshes.append(o.vertices.numpy());joints.append(z['joints_zup_m'])
   midz=(min(v[:,:,2].min() for v in meshes)+max(v[:,:,2].max() for v in meshes))/2;samples=set(np.linspace(0,n-1,8,dtype=int));keys=[]
   for i in range(n):
    image=Image.new('RGB',(960,476),'#f4f7fa');draw=ImageDraw.Draw(image)
    for side in [0,1]:
     if body is not None:scene.remove_node(body)
     body=scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=meshes[side][i],faces=model.faces,process=False),material=pyrender.MetallicRoughnessMaterial(baseColorFactor=([.2,.48,.7,1] if side==0 else [.85,.46,.15,1]),roughnessFactor=.8),smooth=True))
     target=[joints[side][i,0,0],joints[side][i,0,1],midz];mat=cam(target);scene.set_pose(cn,mat);scene.set_pose(ln,mat);rgb,_=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL);image.paste(Image.fromarray(rgb),(480*side,76));draw.text((480*side+10,7),('Original task-selected generator' if side==0 else 'ONE shared generator'),font=F,fill='#17334b');q=pair[side]['human'];draw.text((480*side+10,30),f'{task} | cmd={r["command"]:.3f} | t={i/20:.2f}s',font=F,fill='#17334b');draw.text((480*side+10,52),f'Q={q["quantity"]:.3f} | event={q["event_pass"]}',font=F,fill='#17334b')
    writer.append_data(np.asarray(image))
    if i in samples:frames.append(image.resize((720,357)))
   records.append(dict(task=task,source=r['source'],command=r['command'],baseline=pair[0]['path'],candidate=r['path']))
 sheet=Image.new('RGB',(1440,357*((len(frames)+1)//2)),'white')
 for k,im in enumerate(frames):sheet.paste(im,((k%2)*720,(k//2)*357))
 sheet.save(out/(task+'_inspection.jpg'));print(task,flush=True)
renderer.delete();(out/'protocol.json').write_text(json.dumps(dict(fps=20,speed=1,selection='first prompt/noise,low/mid/high; no success filtering',scope='SMPL-H kinematic reference only. Exact stored pose and root translation. Neutral fingers. No grounding or pose edits. Camera follows pelvis.',records=records),indent=2))
