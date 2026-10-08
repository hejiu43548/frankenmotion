from pathlib import Path
import subprocess,json,time
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';name='demo_inference_20261008';folder=D/'table'/name;out=folder/'videos';out.mkdir(exist_ok=True);summary=folder/'composed/shared_tracker_summary.json'
while True:
 if summary.exists():
  s=json.loads(summary.read_text())
  if s.get('processed')==4:break
 time.sleep(5)
for i in range(4):
 with (out/f'scene_{i:03d}.log').open('w') as f:
  subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/turn_render_20261005.py'),'--run',str(folder/f'composed/scene_{i:03d}/shared_tracker'),'--output',str(out/f'scene_{i:03d}.mp4'),'--label','Unified FrankenMotion | Table interaction'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('RENDERED4',flush=True)
