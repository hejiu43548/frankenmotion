import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,random,csv,time,html,hashlib
from pathlib import Path
import numpy as np,torch,trimesh,pyrender,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');sys.path.insert(0,str(R));os.chdir(R)
from src.tools.inference import load_smplh,extract_motion_outputs
torch.set_num_threads(2)
OUT=R/'outputs_amass/subset_review_200';OUT.mkdir(exist_ok=True)
D=R/'datasets/annotations/frankenstein-category-v1/annotations';data=json.loads((D/'annotations.json').read_text());official=json.loads((R/'datasets/annotations/frankenstein-dataset/annotations/annotations.json').read_text());train=set((D/'splits/train.txt').read_text().split())
seed=20260916;rng=random.Random(seed);items=[]
parts=['action','trajectory','spine','head','left_arm','right_arm','left_leg','right_leg']
for cat in ['walk','run','wave','throw']:
 pool=sorted(k for k,v in data.items() if k in train and v['primary']==cat)
 for i,k in enumerate(rng.sample(pool,50),1):
  v=data[k];sid=v['source_id'];raw=official[sid];assert raw['path']==v['path']
  def relative(aa):
   return [dict(a,source_start=a['start'],source_end=a['end'],start=round(max(a['start'],v['start'])-v['start'],6),end=round(min(a['end'],v['end'])-v['start'],6)) for a in aa if a['end']>v['start'] and a['start']<v['end']]
  stem=f'{i:02d}_{k}';folder=OUT/cat;folder.mkdir(exist_ok=True)
  rec={'id':k,'primary':cat,'split':'train','source_id':sid,'source_path':v['path'],'source_start':v['start'],'source_end':v['end'],'start_frame':v['start_frame'],'end_frame':v['end_frame'],'fps':20,'video':f'{cat}/{stem}.mp4','labels':f'{cat}/{stem}.json','category_annotations':relative(v['annotations']),'official_annotations':relative(raw['annotations']),'original_subset_record':v,'original_official_record':raw}
  (OUT/rec['labels']).write_text(json.dumps(rec,ensure_ascii=False,indent=2));items.append(rec)
manifest={'seed':seed,'sampled_from':'train only; uniform without replacement within primary class','categories':{c:50 for c in ['walk','run','wave','throw']},'items':items}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
with (OUT/'labels.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['id','primary','video','annotation_type','bodypart','text','clip_start_s','clip_end_s','source_start_s','source_end_s','confidence'])
 for x in items:
  for kind in ['category_annotations','official_annotations']:
   for a in x[kind]:w.writerow([x['id'],x['primary'],x['video'],kind,a['bodypart'],a['text'],a['start'],a['end'],a['source_start'],a['source_end'],a.get('confidence','')])
page='''<!doctype html><meta charset="utf-8"><title>Subset 200 review</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;background:#eee}article{background:white;padding:20px;margin:20px 0}video{width:800px;max-width:100%}pre{white-space:pre-wrap}select{font-size:18px}</style><h1>训练子集审核：真实 SMPL 动作，200 条</h1><p>每类随机 50 条，seed=20260916。不是生成结果。20 FPS，两个视角，镜头跟随人体平移；身体左右按解剖学定义。视频底部显示当前训练类别，unknown 表示未标注。原始位置与根轨迹保留在动作中，跟随镜头会减弱整体位移观感。</p><p>标签时间以裁剪视频起点为 0 秒。原始时间与完整官方记录见 JSON。主类别只是抽样分组，不代表每个部位或整段时间都执行该动作。</p><a href="labels.csv">标签 CSV</a> · <a href="manifest.json">抽样清单</a><p>筛选：<select onchange="document.querySelectorAll('article').forEach(x=>x.hidden=this.value!='all'&&x.dataset.cat!=this.value)"><option>all</option><option>walk</option><option>run</option><option>wave</option><option>throw</option></select></p>'''
for x in items:
 page+=f'<article data-cat="{x["primary"]}"><h2>{x["primary"]} / {x["id"]}</h2><p>{html.escape(x["source_path"])} | 原动作 {x["source_start"]}–{x["source_end"]}s</p><video controls preload="none" src="{x["video"]}"></video><p><a href="{x["labels"]}">完整对应标签 JSON</a></p>'
 for key,title in [('category_annotations','训练类别标签'),('official_annotations','官方原文标签')]:
  page+=f'<details {"open" if key=="category_annotations" else ""}><summary>{title}</summary><pre>'+html.escape('\n'.join(f'{a["start"]:.2f}–{a["end"]:.2f}s  {a["bodypart"]}: {a["text"]}' for a in x[key]))+'</pre></details>'
 page+='</article>'
(OUT/'index.html').write_text(page)
smpl=load_smplh(gender='neutral');faces=np.load(R/'src/renderer/humor_render_tools/smplh.faces')
renderer=pyrender.OffscreenRenderer(400,400)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(.48,.66,.82,1),metallicFactor=0,roughnessFactor=.8)
def camera(eye,target):
 z=eye-target;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x);m=np.eye(4);m[:3,:3]=np.stack([x,y,z],1);m[:3,3]=eye;return m
start=time.time();done=[]
for idx,x in enumerate(items):
 dest=OUT/x['video'];n=x['end_frame']-x['start_frame']
 if dest.exists() and dest.stat().st_size>1000:
  done.append(x['id']);continue
 feats=np.load(R/'datasets/motions/AMASS_20.0_fps_nh_smplrifke'/(x['source_path']+'.npy'),mmap_mode='r')[x['start_frame']:x['end_frame']].copy();assert len(feats)==n
 with torch.no_grad():outputs=extract_motion_outputs(torch.from_numpy(feats).float(),n,205,'smplrifke',20,'smpl',smpl)
 verts=outputs['vertices'];joints=outputs['joints'];assert np.isfinite(verts).all()
 ground=float(verts[:,:,2].min());verts[:,:,2]-=ground;joints[:,:,2]-=ground
 temp=dest.with_name(dest.stem+'.tmp.mp4')
 with imageio.get_writer(temp,fps=20,codec='libx264',quality=7,macro_block_size=2,ffmpeg_params=['-threads','2','-movflags','+faststart']) as writer:
  for frame in range(n):
   root=joints[frame,0];center=np.array([root[0],root[1],1.0]);mesh=pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=verts[frame],faces=faces,process=False),material=material,smooth=True)
   views=[]
   for offset in [np.array([3.,-4.,2.0]),np.array([-4.,-2.,1.7])]:
    scene=pyrender.Scene(bg_color=[.96,.96,.96,1],ambient_light=[.5,.5,.5]);scene.add(mesh)
    floor=trimesh.creation.box(extents=[6,6,.015]);floor.apply_translation([root[0],root[1],-.018]);scene.add(pyrender.Mesh.from_trimesh(floor,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(.8,.8,.8,1))))
    pose=camera(center+offset,center);scene.add(pyrender.PerspectiveCamera(yfov=np.pi/5),pose=pose);scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=pose)
    color,_=renderer.render(scene);views.append(color)
   canvas=Image.new('RGB',(800,584),'white');canvas.paste(Image.fromarray(views[0]),(0,0));canvas.paste(Image.fromarray(views[1]),(400,0));draw=ImageDraw.Draw(canvas)
   draw.text((10,5),f'{x["primary"]} / {x["id"]} | t={frame/20:.2f}s',fill='black',font=font)
   for pi,p in enumerate(parts):
    active=[a['text'] for a in x['category_annotations'] if a['bodypart']==p and a['start']<=frame/20<a['end']];draw.text((10,405+pi*21),f'{p}: {", ".join(active) if active else "unknown"}',fill='black',font=font)
   writer.append_data(np.asarray(canvas))
   if idx==0 and frame==n//2:canvas.save(OUT/'preview.jpg')
 temp.replace(dest);done.append(x['id']);print(json.dumps({'done':len(done),'total':200,'video':x['video'],'elapsed_seconds':time.time()-start}),flush=True)
 (OUT/'progress.json').write_text(json.dumps({'done':len(done),'total':200,'elapsed_seconds':time.time()-start}))
renderer.delete();(OUT/'completed.json').write_text(json.dumps({'count':len(done),'seconds':time.time()-start,'seed':seed}));print('COMPLETE',flush=True)
