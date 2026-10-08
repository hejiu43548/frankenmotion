import os
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import json,torch,numpy as np,smplx,pyrender,trimesh,imageio.v2 as io
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/temporal_mix_20261008';G=R/'outputs_amass/unified_commands_20261007/generation/temporal_mix_emphasis_20261008';out=D/'videos_emphasis';out.mkdir(exist_ok=True);torch.set_num_threads(1);model=smplx.SMPLH(str(R/'deps/smplh/SMPLH_MALE.npz'),ext='npz',use_pca=False,flat_hand_mean=True,num_betas=10).eval();renderer=pyrender.OffscreenRenderer(640,540);scene=pyrender.Scene(bg_color=[.92,.94,.96,1],ambient_light=[.5,.5,.5]);floor=trimesh.creation.box(extents=[30,30,.015]);floor.apply_translation([0,0,-.02]);scene.add(pyrender.Mesh.from_trimesh(floor,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.58,.63,.68,1],roughnessFactor=1)));cn=scene.add(pyrender.PerspectiveCamera(yfov=np.pi/4));ln=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=3));node=None;F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
def camera(target):
 eye=np.array(target)+np.array([2.8,-3.2,1.15]);back=eye-target;back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);m=np.eye(4);m[:3,:3]=np.stack([right,np.cross(back,right),back],1);m[:3,3]=eye;return m
for seed in range(1):
 zs=[np.load(G/f'{mode}_s{seed}.npz') for mode in ['text_only','speed_0p6']];meshes=[]
 for z in zs:
  p=torch.tensor(z['poses_axisangle']).float();n=len(p)
  with torch.no_grad():v=model(global_orient=p[:,:3],body_pose=p[:,3:],transl=torch.tensor(z['root_translation']).float(),betas=torch.zeros(n,10),left_hand_pose=torch.zeros(n,45),right_hand_pose=torch.zeros(n,45));assert np.max(abs(v.joints.numpy()[:,:22]-z['joints_zup_m'][:,:22]))<.001
  meshes.append(v.vertices.numpy())
 samples=[]
 with io.get_writer(str(out/f'human_comparison_s{seed}.mp4'),fps=20,codec='libx264',quality=8,macro_block_size=1) as writer:
  for i in range(n):
   im=Image.new('RGB',(1280,640),'#f4f7fa');draw=ImageDraw.Draw(im);t=i/20
   for side in range(2):
    if node is not None:scene.remove_node(node)
    node=scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=meshes[side][i],faces=model.faces,process=False),material=pyrender.MetallicRoughnessMaterial(baseColorFactor=([.2,.48,.7,1] if side==0 else [.85,.46,.15,1]),roughnessFactor=.8),smooth=True));root=zs[side]['joints_zup_m'][i,0];mat=camera([root[0],root[1],.9]);scene.set_pose(cn,mat);scene.set_pose(ln,mat);rgb,_=renderer.render(scene);im.paste(Image.fromarray(rgb),(640*side,100));draw.text((640*side+12,8),'Text timeline only' if side==0 else 'Text timeline + speed 0.6 m/s',font=F,fill='#17334b');draw.text((640*side+12,34),f'Human reference | seed {10808100+seed} | motion t={t:.2f}s',font=F,fill='#17334b')
   wanted=f'REQUEST: Walk 1-5s [{"ON" if 1<=t<5 else "off"}] | Left wave 2-4s [{"ON" if 2<=t<4 else "off"}] | Right wave 3-5s [{"ON" if 3<=t<5 else "off"}]';draw.text((12,66),wanted,font=F,fill='#17334b');writer.append_data(np.asarray(im))
   if i in [10,30,45,55,65,75,85,95,110]:samples.append(im.resize((768,384)))
 sheet=Image.new('RGB',(1536,384*5),'white')
 for k,img in enumerate(samples):sheet.paste(img,((k%2)*768,(k//2)*384))
 sheet.save(out/f'human_comparison_s{seed}.jpg');print(seed,flush=True)
renderer.delete()
