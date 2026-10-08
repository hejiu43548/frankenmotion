from pathlib import Path
import json,hashlib,imageio.v2 as imageio,datetime
D=Path('/home/pku/frankenmotion/outputs_amass/unified_commands_20261007');audit=D/'visual_review/video_decode_audit.json';rows=json.loads(audit.read_text())
for f in sorted((D/'visuals/failure_pairs').glob('*.mp4')):
 rd=imageio.get_reader(f);meta=rd.get_meta_data();n=0
 for frame in rd:n+=1
 rd.close();assert n>1;rows=[r for r in rows if r['file']!=str(f)];rows.append(dict(file=str(f),frames=n,fps=meta['fps'],duration=n/meta['fps'],shape=list(frame.shape)))
assert len(rows)==len(list((D/'visuals').rglob('*.mp4')))==51;audit.write_text(json.dumps(rows,indent=2))
backup=json.loads((D/'backup_manifest.json').read_text());print('BACKUP_SCHEMA',type(backup).__name__)
# Record trial metadata and paths without modifying any existing baseline files.
source=D/'exports/candidate_v4/unified_generator.pt';lock=json.loads((D/'selection_lock.json').read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['sha256'];s=json.loads((D/'report/summary.json').read_text());assert json.loads((D/'report/acceptance_checks.json').read_text())['near_baseline'];(D/'delivery_audit.json').write_text(json.dumps(dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=lock['sha256'],video_count=len(rows),evaluated_requests_per_variant=s['requests'],final_training_after_selection=False,original_defaults_replaced=False,pushed=False),indent=2));print('DELIVERY_AUDIT_PASS',len(rows))
