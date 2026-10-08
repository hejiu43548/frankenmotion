"""Verify delivered encoded videos decode, and match their recorded hashes."""
from pathlib import Path
import json,hashlib,subprocess,datetime
import imageio_ffmpeg,imageio.v2 as imageio
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');index=json.loads((D/'video_index.json').read_text());ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();results=[];previous=D/'media_integrity.json';cache={r['video']:r for r in json.loads(previous.read_text())['videos']} if previous.exists() else {}
for row in index['videos']+index.get('diagnostic_videos',[]):
 suffix=row['local_path'].split('/outputs/universal_tracker_20261006/',1)[1];p=D/suffix;sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==row['sha256'],p
 if suffix in cache and cache[suffix]['sha256']==sha:
  results.append(cache[suffix]);print(suffix,'unchanged; prior decode verified',flush=True);continue
 proc=subprocess.run([ffmpeg,'-v','error','-threads','2','-i',str(p),'-f','null','-'],capture_output=True,text=True);assert proc.returncode==0,(p,proc.stderr)
 reader=imageio.get_reader(str(p));meta=reader.get_meta_data();count=reader.count_frames();reader.close();assert abs(meta['fps']-25)<1e-6;results.append(dict(video=suffix,sha256=sha,decode_returncode=proc.returncode,decoder_messages=proc.stderr,frames=count,fps=meta['fps'],duration=meta['duration'],size=meta['size']));print(suffix,count,flush=True)
(D/'media_integrity.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,videos=results),indent=2))
