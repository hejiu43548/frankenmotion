"""Snapshot reproducible experiment code and frozen adapters, never mutate baseline."""
from pathlib import Path
import json,shutil,hashlib,tarfile
ROOT=Path('/home/pku/frankenmotion');NEW=ROOT/'outputs_amass/franken_improve_20261003';BASE=ROOT/'outputs_amass/franken_eleven_20261003';DEST=NEW/'reproduction_v3';DEST.mkdir(exist_ok=True)
# Final packaging must not silently omit unfinished required results.
read=lambda p:json.loads(p.read_text())
required=[NEW/'final_delivery_capacity256/trajectory_audit.json',NEW/'final_delivery_capacity256/protocol_audit.json',NEW/'final_delivery_integrated_v3_capacity256/audit.json',NEW/'final_delivery_novel_v3_capacity256/audit.json',NEW/'final_delivery_specialists/selection.json',NEW/'capacity_impact_v1.json',NEW/'reports/整合V3最终报告.txt']
for p in required:assert p.is_file(),str(p)
a=read(required[0]);assert a['records_checked']==3520 and a['raw_recalculation_max_abs_error']<1e-8 and a['overflow_warnings']==0
assert read(required[1])['status']=='passed'
a=read(required[2]);assert a['checks']==1120 and a['frozen_hashes_verified'] and a['raw_recalc_max_error']<1e-8 and a['overflow_warnings']==0
a=read(required[3]);assert a['requests']==110 and a['raw_max_error']<1e-8 and a['overflow_warnings']==0
assert all(not r['passes_preregistered_gate'] for r in read(required[4])['results'].values()),'Selected specialist needs frozen holdout before final packaging'
items=list((ROOT/'work').glob('*_20261003.py'))+list((BASE/'code').glob('*.py'))+list((NEW/'frozen_generation_v1').glob('*'))+list((NEW/'frozen_controllers_v1').glob('*'))
items+=list((ROOT/'outputs_amass/root_control_20260925/code').glob('*.py'))
items += [NEW/'asset_provenance.json',NEW/'marker_geometry_audit.json',BASE/'skeleton.npz',BASE/'evaluation_manifest.json',BASE/'data_manifest.json',BASE/'task_adapter_deploy.pt',ROOT/'outputs_amass/root_control_20260925/base_config.yaml']
items += list((BASE/'prompts').glob('*.pt'))+list((BASE/'prompts').glob('*.txt'))
items += [NEW/(x+'_environment.json') for x in ['generator','sonic','mjlab']]
items += [NEW/'candidate_confirmation_manifest.json',NEW/'paired_baseline_confirmation_manifest.json']
items += list((NEW/'final_delivery').glob('*'))
for name in ['final_delivery_v2','final_delivery_kick_v3','final_delivery_integrated_v3','final_delivery_integrated_v3_capacity256','final_delivery_capacity256','final_delivery_novel_v3_capacity256','frozen_retarget_v2','frozen_kick_v3','frozen_integrated_v3']:
 items += [p for p in (NEW/name).glob('*') if p.is_file()]
items += list(NEW.glob('integrated_v3*manifest.json'))
items += list(NEW.glob('kick_v3*manifest.json'))
items += [NEW/'capacity_correction_protocol.json',NEW/'asset_provenance_v3.json',NEW/'turn_timing_audit.json',NEW/'turn_settle_development/results.json']
items += [NEW/'retarget_v2_confirmation_manifest.json',NEW/'specialist_selection_rule.json']
items += list(NEW.glob('capacity_impact_*.json'))
items += [ROOT/'work'/'finish_capacity_retry.log']
for name in ['specialist_turn_development_capacity256','specialist_strike_development_capacity256','specialist_dance_control_capacity256','final_delivery_specialists']:
 items += [p for p in (NEW/name).glob('*') if p.is_file() and p.suffix in ['.json','.csv','.txt']]

items += [ROOT/'work'/name for name in ['capacity_integrated_v3.log','capacity_novel_v3.log','capacity_v1.log','recheck_capacity.log','finish_capacity.log']]
items += [NEW/'novel_prompt_v3/protocol.json',NEW/'novel_prompt_v3/manifest.json']
items += [p for p in (NEW/'novel_prompt_v3/prompts').glob('*') if p.is_file()]


for name in ['beyondmimic_finetune_turn_all_ori2.0_vel1.0','beyondmimic_finetune_strike_all_ori0.5_vel2.0']:
 items += [NEW/name/'model_2399.pt']
items += [ROOT/'work'/name for name in ['specialist_turn.log','specialist_strike.log','finish_specialists.log']]
items += list((NEW/'reports').glob('*.txt'))
items += list((NEW/'confirmation_videos').glob('metadata.json'))
items += list(NEW.glob('*development*manifest.json'))+list(NEW.glob('*development*results.json'))
for folder in NEW.iterdir():
 if folder.is_dir() and (folder.name.startswith('beyondmimic_') or folder.name in ['physical_adapter','jump_aligned_adapter','jump_imitation_adapter']):
  items += [folder/name for name in ['protocol.json','complete.json','status.json','results.json'] if (folder/name).exists()]
  if folder.name.startswith('beyondmimic_finetune_'):items += list(folder.glob('training_motions.npz'))

for name in ['smoke_wave_v2','smoke_side','smoke_v2_raise','smoke_v2_lean','smoke_v3_kick','smoke_v3_capacity_side']:
 items += list((NEW/'single_requests'/name).glob('*.json'))
records=[]
for src in items:
 if not src.is_file():continue
 rel=src.relative_to(ROOT);dst=DEST/'project_overlay'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);records.append(dict(path=str(Path('project_overlay')/rel),original=str(src),bytes=src.stat().st_size,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
(DEST/'files.json').write_text(json.dumps(records,indent=2))
(DEST/'README.txt').write_text('''FrankenMotion G1 integrated experiment snapshot v3

Latest integrated results are in final_delivery_integrated_v3_capacity256, while final_delivery
is the preserved first-round V1 benchmark. The Chinese V3 report identifies
remaining failures; no hardware deployment is included.

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
.conda/bin/python work/run_command_v3_20261003.py --task wave --command 0.15 --seed 91023001 --name unique_wave_name
.conda/bin/python work/run_command_v3_20261003.py --task sidestep --command 0.8 --seed 91023001 --name unique_side_name

Use fresh output names. See Chinese single-command instructions for units/ranges.
Only four cached text templates per task are exposed by this entrypoint.
Frozen controller protocol records development-only selection and SHA256 hashes.
No seamless controller transition or hardware behavior has been evaluated.

Regenerate audited metrics and figures AFTER all independent rollouts are present:
work/g1_sim_env/bin/python work/assess_integrated_v3_capacity_20261003.py
PYTHONPATH=outputs_amass/g1_command_diagnosis_20261002_2346/plot_deps work/g1_sim_env/bin/python work/plot_integrated_v3_capacity_20261003.py

The frozen batch driver uses fixed archive paths. Do not rerun it over retained
benchmark data; reproduce in an isolated project/output copy with paths adjusted.
For normal testing use the single-request entrypoint with a fresh --name.

Full experiment stages and CLI flags are in generate_physical, confirmation_eval,
mjlab_probe, physical_adapter, jump_aligned_adapter, mjlab_finetune. Protocols,
manifest seeds and all failure records are preserved. Do not re-select controllers
on the existing confirmation results; new selection requires a fresh holdout.
''')
archive=NEW/'frankenmotion_g1_reproduction_v3.tar.gz'
with tarfile.open(archive,'w:gz') as f:f.add(DEST,arcname=DEST.name)
print(json.dumps(dict(archive=str(archive),files=len(records),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
