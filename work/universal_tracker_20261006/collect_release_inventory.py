"""Collect immutable provenance and runnable source without publishing or deleting experiments."""
import json,hashlib,subprocess,tarfile,datetime,sys,shutil
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
release=D/'release';release.mkdir(exist_ok=True);records=[]
for folder in sorted((D/'training').iterdir()):
 if not folder.is_dir():continue
 protocol=folder/'protocol.json';checkpoints=[]
 for ready in sorted(folder.glob('*.pt.ready.json')):
  r=json.loads(ready.read_text());checkpoints.append(dict(ready_file=str(ready),metadata=r))
 completed=folder/'complete.json'
 records.append(dict(name=folder.name,protocol=json.loads(protocol.read_text()) if protocol.exists() else None,complete=completed.exists(),completion=json.loads(completed.read_text()) if completed.exists() else None,checkpoints=checkpoints))
sources=sorted(set(W.glob('*.py'))|set((R/'work').glob('*2026100*.py'))|set((R/'outputs_amass/franken_eleven_20261003/code').glob('*.py')))
with tarfile.open(release/'source_snapshot.tar.gz','w:gz') as archive:
 for f in sources:archive.add(f,arcname=str(f.relative_to(R)))
 for pattern in ['*.md','*.json']:
  for f in W.glob(pattern):archive.add(f,arcname=str(f.relative_to(R)))
shutil.copy2(W/'metric_definitions.json',release/'metric_definitions.json')
backup_expected={'stable_frozen/policy.pt':'64eb503f97fb426d8b54fca2c363803a9c18695b5c1c61b1ac5ebeb4246f6aaf','prior_unified_frozen/policy.pt':'72218056174c917a064ccf2c0a15e24601f9ed8df17cded6841be8f73936c217'}
backup_check={name:sha(D/'backup'/name) for name in backup_expected};assert backup_check==backup_expected
frozen=json.loads((D/'frozen_unified/protocol.json').read_text());assert sha(Path(frozen['checkpoint']))==frozen['checkpoint_sha256']
contract=json.loads((D/'table_evaluation/baseline_stable/scene_000/reference_input/inference_contract.json').read_text());contract.pop('initial_qpos',None);contract.pop('initial_qvel',None);contract['preview_offsets']=frozen['preview_offsets'];contract['checkpoint_sha256']=frozen['checkpoint_sha256'];contract['actor_sha256']=frozen['actor_sha256'];contract['scope']='Native simulation inference interface; initialization depends on each scene. Normalization embedded in actor. Requires matching model, sensors, joint/action ordering and reference body indexing; not validated on hardware.'
(release/'inference_contract.json').write_text(json.dumps(contract,indent=2))
flat_model=D/'general_evaluation/fresh_candidate/scene.mjb'
if flat_model.exists():shutil.copy2(flat_model,release/'native_nominal_scene.mjb')
versions=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout;(release/'python_environment.txt').write_text(versions)
git=subprocess.run(['git','rev-parse','HEAD'],cwd=R,capture_output=True,text=True);status=subprocess.run(['git','status','--short'],cwd=R,capture_output=True,text=True)
(release/'git_status.txt').write_text(status.stdout)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True);(release/'gpu_hardware.txt').write_text(gpu.stdout)
inventory=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=str(R),git_head=git.stdout.strip(),frozen=frozen,backup_sha256=backup_check,training=records,source_sha256={str(f.relative_to(R)):sha(f) for f in sources},source_archive_sha256=sha(release/'source_snapshot.tar.gz'),python_executable=sys.executable,environment_sha256=sha(release/'python_environment.txt'),scope='Source and provenance snapshot. Requires original repository assets, model weights and the recorded environment on the remote server; not a standalone self-contained model download. No push performed.')
(release/'inventory.json').write_text(json.dumps(inventory,indent=2));print(release)
