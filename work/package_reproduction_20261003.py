"""Snapshot reproducible experiment code and frozen adapters, never mutate baseline."""
from pathlib import Path
import json,shutil,hashlib,tarfile
ROOT=Path('/home/pku/frankenmotion');NEW=ROOT/'outputs_amass/franken_improve_20261003';BASE=ROOT/'outputs_amass/franken_eleven_20261003';DEST=NEW/'reproduction_v1';DEST.mkdir(exist_ok=True)
items=list((ROOT/'work').glob('*_20261003.py'))+list((BASE/'code').glob('*.py'))+list((NEW/'frozen_generation_v1').glob('*'))+list((NEW/'frozen_controllers_v1').glob('*'))
items+=list((ROOT/'outputs_amass/root_control_20260925/code').glob('*.py'))
items += [NEW/'asset_provenance.json',NEW/'marker_geometry_audit.json',BASE/'skeleton.npz',BASE/'evaluation_manifest.json',BASE/'data_manifest.json',BASE/'task_adapter_deploy.pt',ROOT/'outputs_amass/root_control_20260925/base_config.yaml']
items += list((BASE/'prompts').glob('*.pt'))+list((BASE/'prompts').glob('*.txt'))
items += [NEW/(x+'_environment.json') for x in ['generator','sonic','mjlab']]
items += [NEW/'candidate_confirmation_manifest.json',NEW/'paired_baseline_confirmation_manifest.json']
items += list((NEW/'final_delivery').glob('*'))
items += list((NEW/'reports').glob('*.txt'))
items += list((NEW/'confirmation_videos').glob('metadata.json'))
items += list(NEW.glob('*development*manifest.json'))+list(NEW.glob('*development*results.json'))
for folder in NEW.iterdir():
 if folder.is_dir() and (folder.name.startswith('beyondmimic_') or folder.name in ['physical_adapter','jump_aligned_adapter','jump_imitation_adapter']):
  items += [folder/name for name in ['protocol.json','complete.json','status.json','results.json'] if (folder/name).exists()]
  if folder.name.startswith('beyondmimic_finetune_'):items += list(folder.glob('training_motions.npz'))

for name in ['smoke_wave_v2','smoke_side']:
 items += list((NEW/'single_requests'/name).glob('*.json'))
records=[]
for src in items:
 if not src.is_file():continue
 rel=src.relative_to(ROOT);dst=DEST/'project_overlay'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);records.append(dict(path=str(Path('project_overlay')/rel),original=str(src),bytes=src.stat().st_size,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
(DEST/'files.json').write_text(json.dumps(records,indent=2))
(DEST/'README.txt').write_text('''FrankenMotion G1 experiment snapshot v1

This is a project overlay for the existing /home/pku/frankenmotion checkout.
It includes experiment scripts, frozen task adapters/controllers, prompt caches,
source metadata, per-request metrics and environment/provenance manifests.
It does not include the multi-GB base checkpoint, root adapter, vendor repositories,
SONIC ONNX models, robot mesh assets, training dataset or thousands of raw rollouts.
Those dependencies remain on the authorized server; their locations/hashes are
recorded in asset_provenance.json. This is NOT a standalone all-dependencies bundle.

Do not overwrite the original baseline when inspecting this archive. On another
machine reconstruct the pinned project/dependencies first and adjust hard-coded
/home/pku/frankenmotion paths consistently. Python environments are deliberately
separate: generator .conda/bin/python; SONIC work/g1_sim_env/bin/python;
BeyondMimic work/mjlab_stable_env/bin/python. Environment manifests record versions.

Single request (server already configured):
cd /home/pku/frankenmotion
.conda/bin/python work/run_command_20261003.py --task wave --command 0.15 --seed 91023001 --name unique_wave_name
.conda/bin/python work/run_command_20261003.py --task sidestep --command 0.8 --seed 91023001 --name unique_side_name

Use fresh output names. See Chinese single-command instructions for units/ranges.
Only four cached text templates per task are exposed by this entrypoint.
Frozen controller protocol records development-only selection and SHA256 hashes.
No seamless controller transition or hardware behavior has been evaluated.

Regenerate audited metrics and figures AFTER all independent rollouts are present:
work/g1_sim_env/bin/python work/recheck_trajectories_20261003.py
work/g1_sim_env/bin/python work/assess_confirmation_20261003.py
PYTHONPATH=outputs_amass/g1_command_diagnosis_20261002_2346/plot_deps work/g1_sim_env/bin/python work/plot_confirmation_20261003.py

Full experiment stages and CLI flags are in generate_physical, confirmation_eval,
mjlab_probe, physical_adapter, jump_aligned_adapter, mjlab_finetune. Protocols,
manifest seeds and all failure records are preserved. Do not re-select controllers
on the existing confirmation results; new selection requires a fresh holdout.
''')
archive=NEW/'frankenmotion_g1_reproduction_v1.tar.gz'
with tarfile.open(archive,'w:gz') as f:f.add(DEST,arcname=DEST.name)
print(json.dumps(dict(archive=str(archive),files=len(records),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
