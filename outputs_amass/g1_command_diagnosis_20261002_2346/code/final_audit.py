import json,os,hashlib,subprocess
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image,ImageDraw
OUT=Path(os.environ['DIAG_OUT']);DEL=OUT/'delivery'
paired=json.loads((OUT/'paired_check/results.json').read_text())
summary=json.loads((DEL/'summary.json').read_text())
summary.update(paired_previous_success=sum(r['previous_success'] for r in paired),paired_shared_success=sum(r['success'] for r in paired),paired_total=len(paired),
               paired_shared_mae_mps=float(np.mean([abs(r['speed_mps']-r['command_mps']) for r in paired])))
(DEL/'summary.json').write_text(json.dumps(summary,indent=2))
raw=[]
for p in sorted(OUT.rglob('rollouts/*.npz')):
    z=np.load(p);m=json.loads(p.with_suffix('.json').read_text());q=z['qpos'];err=None
    if m['complete']:
        from scipy.spatial.transform import Rotation
        axis=Rotation.from_quat(z['reference_qpos'][50,[4,5,6,3]]).apply([1.,0,0])[:2]
        speed=float((q[-1,:2]-q[50,:2])@axis/((len(q)-51)/50))
        err=abs(speed-m['speed_mps']);assert err<1e-12
    raw.append(dict(path=str(p.relative_to(OUT)),frames=len(q),complete=m['complete'],metric_error=err,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(raw)==323,len(raw)
demo=np.load(OUT/'demos/delivery_replay/rollouts/demo.npz')['qpos']
original=np.load(OUT/'rollouts/shared_04_walk_wave_v3_v0.45_s6613.npz')['qpos']
replay_error=float(np.max(np.abs(demo-original)));assert replay_error==0
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg,'-v','error','-i',str(DEL/'g1_command_response.mp4'),'-f','null','-'],check=True)
reader=imageio.get_reader(DEL/'g1_command_response.mp4');frames=reader.count_frames();reader.close()
ids=[0,87,173,174,261,347]
# Compact contact sheet of start/middle/end across both displayed source clips.
sheet=Image.new('RGB',(1890,580),(255,255,255))
reader=imageio.get_reader(DEL/'g1_command_response.mp4')
for n,i in enumerate(ids):sheet.paste(Image.fromarray(reader.get_data(i)).resize((630,290)),((n%3)*630,(n//3)*290))
reader.close();sheet.save(DEL/'video_contact_sheet.jpg')
audit=dict(total_new_physical_rollouts=len(raw),complete=sum(r['complete'] for r in raw),
           maximum_raw_metric_error=max(r['metric_error'] or 0 for r in raw),demo_replay_max_state_error=replay_error,
           video_frames=frames,video_full_decode='passed',rollouts=raw,
           code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'code').glob('*.py')})
(DEL/'final_audit.json').write_text(json.dumps(audit,indent=2))
# Account for the 20 validation requests rejected before physics by the first methods.
missing=[dict(case='04_walk_wave_v2',method=method,command_mps=c,seed=s,success=False,reason='No sweep candidate met the predeclared heading gate; no physical validation launched') for method in ['cadence','cadence_amplitude'] for c in [.2,.3,.4,.5,.6] for s in [1301,2702]]
(DEL/'selection_rejections.json').write_text(json.dumps(missing,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k not in ['rollouts','code_sha256']},indent=2))
print(json.dumps(summary,indent=2))
