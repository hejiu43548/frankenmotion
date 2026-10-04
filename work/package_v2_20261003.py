from pathlib import Path
import json,tarfile,hashlib
root=Path('/home/pku/frankenmotion');n=root/'outputs_amass/franken_improve_20261003';items=list((n/'final_delivery_v2').glob('*'))+list((n/'frozen_retarget_v2').glob('*'))+[root/'work'/s for s in ['run_command_v2_20261003.py','generate_v2_20261003.py','assess_retarget_v2_20261003.py','plot_retarget_v2_20261003.py','package_v2_20261003.py']]+[n/'retarget_v2_confirmation_manifest.json',n/'generated/retarget_v2_confirmation/provenance.json']
for name in ['smoke_v2_raise','smoke_v2_lean']:items+=list((n/'single_requests'/name).glob('*.json'))
items=[p for p in items if p.is_file()];records=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in items];meta=n/'v2_supplement_files.json';meta.write_text(json.dumps(dict(note='Supplement to immutable v1 reproduction package; same server dependencies and weights. Overlay preserves relative project paths.',files=records),indent=2));archive=n/'frankenmotion_g1_v2_supplement.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in items:t.add(p,arcname=str(p.relative_to(root)))
 t.add(meta,arcname='v2_supplement_files.json')
print(archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())
