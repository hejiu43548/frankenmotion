"""Freeze generator weights BEFORE independent confirmation metrics are inspected."""
import sys,json,hashlib,shutil,subprocess,datetime
from pathlib import Path
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');folder=ROOT/'frozen_generation_v1';folder.mkdir(exist_ok=True)
choices={'physical':ROOT/'physical_adapter/best.pt','jump_aligned':ROOT/'jump_aligned_adapter/best.pt'};records={}
for key,path in choices.items():
 digest=hashlib.sha256(path.read_bytes()).hexdigest();dest=folder/(key+'.pt')
 if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
 else:shutil.copy2(path,dest)
 records[key]=dict(source=str(path),frozen=str(dest),sha256=digest)
protocol=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),generators=records,routing='jump uses jump_aligned; all other tasks use physical',confirmation_seeds='42026000+prompt_index*100+noise_index',n=880,prompts='same four templates used in development; heldout noise only',selection='development 990700+task, no independent confirmation metrics inspected',controller_selection='still in development, must freeze before inspecting confirmation')
if not (folder/'protocol.json').exists():(folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
py='/home/pku/frankenmotion/.conda/bin/python';script='/home/pku/frankenmotion/work/generate_physical_20261003.py'
for key,label,tasks in [('physical','candidate_nonjump','raise_hand,reach,strike,wave,turn,sidestep,back_walk,kick,lean,walk'),('jump_aligned','candidate_jump','jump')]:
 subprocess.run([py,script,'--phase','confirmation','--weights',records[key]['frozen'],'--label',label,'--tasks',tasks],check=True)
rows=json.loads((ROOT/'candidate_nonjump_confirmation_manifest.json').read_text())+json.loads((ROOT/'candidate_jump_confirmation_manifest.json').read_text());assert len(rows)==880;assert len({(r['source'],r['command_index']) for r in rows})==880
(ROOT/'candidate_confirmation_manifest.json').write_text(json.dumps(rows,indent=2));print('Candidate reference generation complete: 880',flush=True)
