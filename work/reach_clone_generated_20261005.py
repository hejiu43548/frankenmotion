import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();D=Path('/home/pku/frankenmotion/outputs_amass/reach_demo_20261005');src=D/a.source;dst=D/a.destination;dst.mkdir(exist_ok=False);rows=json.loads((src/'manifest.json').read_text())
for row in rows:
 old=Path(row['source']);new=dst/old.name;new.mkdir()
 for f in old.glob('human_*.npz'):shutil.copy2(f,new/f.name)
 row['source']=str(new)
(dst/'manifest.json').write_text(json.dumps(rows,indent=2))
