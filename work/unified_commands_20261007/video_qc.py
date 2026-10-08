from pathlib import Path
import json,imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw
D=Path('/home/pku/frankenmotion/outputs_amass/unified_commands_20261007');out=D/'visual_review';out.mkdir(exist_ok=True);records=[]
for f in sorted((D/'visuals').rglob('*.mp4')):
 reader=imageio.get_reader(f);meta=reader.get_meta_data();n=0;first_shape=None
 for frame in reader:
  assert frame.ndim==3 and frame.shape[-1]==3
  if first_shape is None:first_shape=frame.shape
  assert frame.shape==first_shape;n+=1
 reader.close();assert n>1;records.append(dict(file=str(f),frames=n,fps=meta['fps'],duration=n/meta['fps'],shape=first_shape))
(out/'video_decode_audit.json').write_text(json.dumps(records,indent=2));print('DECODED',len(records))
for f in sorted((D/'visuals/unified_final').glob('*_inspection.jpg')):
 im=Image.open(f);height=im.height//3;im.crop((0,height,im.width,2*height)).save(out/(f.stem+'_middle.jpg'))
# Dense 10Hz approach excerpt and whole-scene 2Hz overview; true saved video frames.
for i in [0,1,2,3]:
 f=D/f'visuals/table/scene_{i:03d}.mp4';reader=imageio.get_reader(f);fps=reader.get_meta_data()['fps'];clips={'approach':np.arange(.5,5.31,.1),'overview':np.arange(0,26,.5)}
 for label,times in clips.items():
  images=[]
  for t in times:
   try:frame=reader.get_data(round(t*fps))
   except IndexError:break
   im=Image.fromarray(frame).resize((320,180));ImageDraw.Draw(im).text((8,95),f'{t:.1f}s',fill='white',stroke_fill='black',stroke_width=1);images.append(im)
  sheet=Image.new('RGB',(1920,180*((len(images)+5)//6)),'white')
  for j,im in enumerate(images):sheet.paste(im,((j%6)*320,j//6*180))
  sheet.save(out/f'table_{i:03d}_{label}.jpg')
 reader.close()
