"""Package the frozen actor, evidence and exact committed source after final audit."""
import datetime,hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';REPO=R/'work/git_publish_g1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (U/'final_pipeline_complete.json').exists()
runtime=read(U/'runtime_versions.json');current_sources={p.name:sha(p) for p in sorted((R/'work').glob('unified_*_20261004.py'))};assert runtime['source_sha256']==current_sources,'Refresh runtime source provenance before packaging'
selected=read(U/'frozen_unified/protocol.json');assert sha(Path(selected['checkpoint']))==selected['checkpoint_sha256']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'Commit final code/evidence documentation first'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip();assert branch=='codex/g1-unified-tracker'
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0];assert remote==commit,'Push verified source before packaging'
for name in ['unified_final','routed_baseline_final']:
 audit=read(U/'evaluation'/name/'audit.json');assert audit['aggregate']['requests']==880 and audit['raw_max_error']<1e-8 and audit['overflow_warnings']==0
challenge=read(U/'evaluation/unified_final_challenges/audit.json');assert challenge['unique_checkpoints']==1 and challenge['raw_max_error']<1e-8
assert sum(v['planned'] for v in challenge['summary'].values())==36
smokes=['frozen_wave_smoke','frozen_walk_smoke']
for name in smokes:
 r=read(U/'single_requests'/name/'result.json');assert r['raw_state_audited'] and r['controller']['checkpoint_sha256']==selected['checkpoint_sha256'] and not r['controller']['task_routing']
# A functional entrypoint smoke need not pass the semantic benchmark. Preserve outcomes.
out=U/'delivery';out.mkdir(exist_ok=False);payload=out/'unified_g1_tracker';payload.mkdir()
def copy(path,relative):
 assert path.is_file(),path
 dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
for name in ['policy.pt','actor.pt','actor.json','protocol.json']:copy(U/'frozen_unified'/name,'weights/'+name)
for name in ['protocol.json','paired_preview_protocol.json','paired_preview_complete.json','additional_snapshot_protocol.json','additional_snapshot_complete.json','final_waiter_replaced.json','final_pipeline_complete.json','runtime_versions.json']:copy(U/name,'provenance/'+name)
for name in ['manifest.json','reference_manifest.json','protocol.json','cohort_audit.json']:copy(U/'final_test'/name,'data/final_test/'+name)
for name in ['manifest.json','protocol.json']:copy(U/'final_test/challenges'/name,'data/final_challenges/'+name)
for name in ['manifest.json','clips.json','audit.json']:copy(U/'training_corpus_augmented'/name,'data/training_corpus/'+name)
for folder in sorted((U/'evaluation').iterdir()):
 if not (folder/'audit.json').exists():continue
 for name in ['audit.json','summary.json','protocol.json','contract.json','audited_results.json']:
  if (folder/name).exists():copy(folder/name,'evaluation/'+folder.name+'/'+name)
for folder in sorted((U/'training').glob('joint_*')):
 for name in ['protocol.json','complete.json','preview_calibration.json','boundary_audit.json']:
  if (folder/name).exists():copy(folder/name,'training/'+folder.name+'/'+name)
 for source in (folder/'source_snapshot').glob('*.py'):copy(source,'training/'+folder.name+'/source_snapshot/'+source.name)
for folder in ['statistics','diagnostics']:
 for path in (U/folder).glob('*.json'):copy(path,folder+'/'+path.name)
for name in smokes:copy(U/'single_requests'/name/'result.json','entrypoint_smokes/'+name+'.json')
for ext in ['png','pdf']:copy(U/'figures'/('final_command_responses.'+ext),'figures/final_command_responses.'+ext)
copy(U/'figures/command_responses.csv','figures/command_responses.csv')
copy(REPO/'experiments/g1_tracking/unified/README.md','README.md')
copy(REPO/'experiments/g1_tracking/unified/RESULTS.md','RESULTS.md')
subprocess.run(['git','archive','--format=tar.gz','--output='+str(payload/'source.tar.gz'),'HEAD'],cwd=REPO,check=True)
manifest=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_commit=commit,branch=branch,remote='https://github.com/hejiu43548/frankenmotion',checkpoint_sha256=selected['checkpoint_sha256'],server_root=str(U),scope='One actor checkpoint and actor-only export with audited benchmark evidence and committed source. External base generator/checkpoints, cached templates, robot assets, third-party environments, training arrays and raw simulation trajectories remain on the configured server; this is not a standalone simulator installer.',files={str(p.relative_to(payload)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(payload.rglob('*')) if p.is_file()})
(payload/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
archive=out/'unified_g1_tracker.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(payload,arcname=payload.name)
with tarfile.open(archive) as tar:
 for name,info in manifest['files'].items():
  f=tar.extractfile(payload.name+'/'+name);assert f is not None;assert hashlib.sha256(f.read()).hexdigest()==info['sha256']
summary=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files_verified=len(manifest['files']),git_commit=commit,selected=selected['selected']);(out/'archive_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
