"""Create a reviewed local evidence bundle, never a claim of standalone deployment."""
from pathlib import Path
import json,hashlib,zipfile,datetime
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';pdf=ROOT/'output/pdf/universal_tracker_20261006/统一Tracker扩展实验报告_中文.pdf';target=ROOT/'output/universal_tracker_20261006_交付包.zip';files={}
def add(path,arc):
 path=Path(path);assert path.is_file(),path;assert '..' not in Path(arc).parts;files[arc]=path
add(pdf,'报告/'+pdf.name)
add(D/'backup/manifest.json','backup/manifest.json')
for name in ['交付说明.txt','演示索引.html','report_data.json','video_index.json','media_integrity.json','single_policy_verification.json','natural_paired_errors.json','raw_demo_integrity.json','local_weight_verification.json']:
 add(D/name,name)
for folder in ['frozen_unified','release','tables','replication_seed6107']:
 root=D/folder
 if root.exists():
  for p in root.rglob('*'):
   if p.is_file():add(p,str(p.relative_to(D)))
for folder in ['final_all','final_upperbody','final_locomotion','final_dynamic','architecture','robustness','expanded_training','root_observability','aerial_dynamics','seed_replication','long_horizon']:
 for p in (D/'figures'/folder).rglob('*'):
  if p.is_file():add(p,str(p.relative_to(D)))
video_index=json.loads((D/'video_index.json').read_text())
for r in video_index['videos']+video_index.get('diagnostic_videos',[]):
 p=Path(r['local_path']);add(p,str(p.relative_to(D)))
 for suffix in ['.json','.png']:
  q=p.with_suffix(suffix)
  if q.exists():add(q,str(q.relative_to(D)))
for rel in (D/'demo_evidence_files.txt').read_text().splitlines():add(D/rel,'selected_raw_evidence/'+rel)
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,limitations='Requires original remote repo, generator weights, robot assets and recorded software environment for full reproduction. Primary deployment weight is frozen_unified; replica is supplementary only.',files={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in sorted(files.items())})
with zipfile.ZipFile(target,'w') as archive:
 for name,p in sorted(files.items()):archive.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix in ['.mp4','.npz','.gz'] else zipfile.ZIP_DEFLATED)
 archive.writestr('包内文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2),compress_type=zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
(D/'delivery_bundle.json').write_text(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(files),scope=__doc__),indent=2));print(target)
