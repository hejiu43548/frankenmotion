import json,shutil,hashlib,tarfile,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';out=D/'release';out.mkdir(exist_ok=False)
def copytree(src,dst):shutil.copytree(src,dst,ignore=shutil.ignore_patterns('*.mjb','scene.zip','*.log','*.mp4','*.png','*.pdf'))
shutil.copytree(D/'frozen',out/'frozen');shutil.copytree(D/'report',out/'report');scripts=out/'scripts';scripts.mkdir()
for f in (R/'work').glob('turn_*_20261005.py'):shutil.copy2(f,scripts/f.name)
for name in ['reach_metrics_20261005.py','reach_render_pair_20261005.py','reach_prepare_walk_20261005.py','reach_prepare_corpus_v3_20261005.py','reach_adapter_v6_20261005.py','table_goal_adapter_20261005.py','physical_adapter_20261003.py','unified_retarget_20261004.py','unified_export_actor_20261004.py','gmr_probe_20261003.py','unified_preview_20261004.py','mjlab_cpu_fk_20261003.py']:shutil.copy2(R/'work'/name,scripts/name)
shutil.copytree(R/'outputs_amass/franken_eleven_20261003/code',out/'generation_core');shutil.copy2(R/'work/g1_sim_bridge/g1_runtime.py',scripts/'g1_runtime.py')
copytree(D/'final12',out/'final12');copytree(D/'final_walk',out/'final_walk');copytree(D/'final_away',out/'final_away');copytree(D/'turn_final',out/'turn_final');training=out/'training';training.mkdir()
for name in ['teacher_v1','full_teacher_v1']:
 p=training/name;p.mkdir()
 for f in ['dataset.npz','reports.json','protocol.json']:shutil.copy2(D/name/f,p/f)
copytree(D/'training_full_v1',training/'references');shutil.copy2(D/'distill_full_v1/progress.json',training/'progress.json');shutil.copy2(D/'distill_full_v1/complete.json',training/'complete.json')
videos=out/'videos';videos.mkdir()
for f in (D/'videos').glob('*.mp4'):shutil.copy2(f,videos/f.name)
shutil.copy2(D/'videos/edit/funding_demo.mp4',videos/'funding_demo.mp4');shutil.copytree(D/'visual_review',out/'visual_review')
example=out/'example/scene_000';example.mkdir(parents=True);scene=D/'final12/scene_000'
for f in ['scene.json','reference_contact.json','reference_contact.npz']:shutil.copy2(scene/f,example/f)
shutil.copytree(scene/'reference_input',example/'reference_input');shutil.copytree(scene/'selected',example/'selected',ignore=shutil.ignore_patterns('scene.mjb'));shutil.copy2(scene/'selected/scene.mjb',example/'selected/scene.mjb')
readme='''FrankenMotion G1 turn demo — reproduction notes

Main videos: videos/funding_demo.mp4, turn_090_full.mp4, turn_135_full.mp4.
report/实验说明.txt describes methods, all twelve outcomes and the one failed135-degree turn. No hardware or Dex3 hand validation is claimed.

Portable CPU replay (Python3.11, torch2.7, mujoco3.5.0, numpy, scipy):
  python scripts/turn_cpu_anchor_evaluate_20261005.py --run example/scene_000/reference_input --actor frozen/actor.pt --checkpoint frozen/policy.pt --output example/scene_000/replay
  python scripts/turn_audit_policy_20261005.py --run example/scene_000/replay --actor frozen/actor.pt
  python scripts/turn_audit_rollout_20261005.py --run example/scene_000/replay
Paths above are relative to this extracted directory. The output directory must not exist. GPU, mjlab and original generator weights are not required for this replay.

The actual controller is one frozen361-input29-output actor. Two causal rigid reference placements anchor turn and departure commands to measured robot pose. initial_motion.npz and anchor_events.json preserve reference causality. No robot state is written after initialization; no state filtering or action-weight switching occurs.

Training datasets contain teacher labels, never deployment outputs. teacher_v1 uses the old smooth and reach actors. full_teacher_v1 adds SONIC supervision for turn/away. Source training and generation scripts retain the original server layout /home/pku/frankenmotion. Full generation/training requires that repository, pretrained backbone, SMPL skeleton, text embeddings, GMR assets, and the documented existing environments; this archive does not pretend to include those large external dependencies. Frozen goal/reach adapters, single actor, CPU scene, generated final references, all final raw trajectories, full-sequence training references and distillation datasets are included.

All twelve test rows, including failed case11, are retained. Per-case MuJoCo binary models except the example are omitted to reduce duplication; native_batch rebuilds table transforms from scene.json on the supplied native model. Illustrative clips are case000 and case003; reach comparison uses002/003. Final tests use new random seeds after freezing the actor.

No push was performed for this experiment. Robot: G1 29DoF body with fixed rubber hand meshes/capsules, not the three-finger Dex3 model.
'''
(out/'README.txt').write_text(readme)
files={str(p.relative_to(out)):dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file()};(out/'manifest_sha256.json').write_text(json.dumps(files,indent=2));archive=D/'frankenmotion_g1_turn_demo.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='frankenmotion_g1_turn_demo')
audit=dict(files=len(files),archive=str(archive),size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),created_unix=time.time());(D/'release_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
