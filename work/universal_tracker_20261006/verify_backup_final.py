"""Verify every originally backed-up artifact remains byte-identical at delivery."""
from pathlib import Path
import json,hashlib,datetime
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');m=json.loads((D/'backup/manifest.json').read_text());failed=[]
for name,expected in m['files'].items():
 p=D/'backup'/name;actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 if actual!=expected:failed.append(dict(path=name,expected=expected,actual=actual))
assert not failed,failed
out=dict(scope=__doc__,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_verified=len(m['files']),all_match=True,manifest_sha256=hashlib.sha256((D/'backup/manifest.json').read_bytes()).hexdigest());(D/'release/backup_final_audit.json').write_text(json.dumps(out,indent=2));print(out)
