"""Post-freeze exploratory reference projection, training sources only, one global teacher.
This experiment never changes the frozen release and never evaluates final test sources.
"""
import json,hashlib,subprocess,os
from pathlib import Path
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python';out=D/'rollout_projection_study';out.mkdir(exist_ok=False)
original=json.loads((D/'joint_corpus_expanded_corrected/manifest.json').read_text());extras=[r for r in original if r['task'].startswith('extra_')];pool=[r for name in ['natural_extended_corrected_motion','natural_locomotion_motion'] for r in json.loads((D/name/'manifest.json').read_text()) if r['split']=='train'];selected=[]
for r in extras:
 matches=[s for s in pool if s['source']==r['source'] and s['task']==r['original_task']];assert len(matches)==1,(r['source'],len(matches));selected.append(matches[0])
assert len(selected)==307;manifest=out/'training_sources.json';manifest.write_text(json.dumps(selected,indent=2));scope=dict(scope=__doc__,teacher=str(D/'frozen_unified/policy.pt'),teacher_sha256=json.loads((D/'frozen_unified/protocol.json').read_text())['checkpoint_sha256'],acceptance='Full RSI rollout, mean global body error <=0.20m, frame-mean error P95<=0.40m. Same horizon. Others retain original reference; every original clip remains. No semantic-class selection.',paired_training='Two 2000-update continuations, seed6108, same frozen v5 initial and configuration; raw references versus accepted rollout replacements. Development-only evaluation. Not a deployment candidate.',main_frozen_test_seen=True,interpretation='Post-test hypothesis exploration, not confirmatory test evidence. References have dynamics of the evaluation model; no proof of cross-model feasibility or preservation of all semantics.')
(out/'protocol.json').write_text(json.dumps(scope,indent=2));subprocess.run([str(py),str(W/'evaluate_cpu_general.py'),'--checkpoint',scope['teacher'],'--manifest',str(manifest),'--name','projection_training_sources','--entry','rsi','--workers','4'],cwd=R,check=True)
results=json.loads((D/'general_evaluation/projection_training_sources/results.json').read_text());lookup={(r['source'],r['task']):r for r in results};corpus=D/'joint_corpus_rollout_projected';corpus.mkdir(exist_ok=False);(corpus/'extra_motions').mkdir();keys=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'];records=[];audit=[];buffers={k:[] for k in keys}
for row in original:
 row=dict(row)
 if row['task'].startswith('extra_'):
  r=lookup[(row['source'],row['original_task'])];accepted=False;p95=None
  if r['physical_complete']:
   run=Path(r['run']);z=np.load(run/'actual.npz');ref=np.load(run/'motion.npz');err=np.linalg.norm(z['body_pos_w']-ref['body_pos_w'],axis=-1).mean(-1);p95=float(np.quantile(err,.95));accepted=float(err.mean())<=.20 and p95<=.40
  audit.append(dict(source=row['source'],task=row['task'],complete=r['physical_complete'],mean_error=r.get('global_mpjpe_m'),p95_error=p95,accepted=accepted))
  if accepted:
   m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);c=json.loads((run/'inference_contract.json').read_text());contract=json.loads((D/'evaluation/baseline_broad_dev/contract.json').read_text());jids=[m.joint('robot/'+n).id for n in contract['joint_names']];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];bids=[m.body('robot/'+n).id for n in contract['body_names']];root=m.body('robot/pelvis').id;logs={k:[] for k in keys}
   for q,v in zip(z['qpos'],z['qvel']):
    d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);xyz=d.xpos[bids].copy();cv=d.cvel[bids];ang=cv[:,:3];lin=cv[:,3:]+np.cross(ang,xyz-d.subtree_com[root])
    for k,value in zip(keys,[q[qa],v[va],xyz,d.xquat[bids],lin,ang]):logs[k].append(value.copy())
   arrays={k:np.asarray(v,dtype=np.float32) for k,v in logs.items()};assert len(arrays['joint_pos'])==row['frames'];assert np.allclose(arrays['body_pos_w'],z['body_pos_w'],atol=1e-6)
   path=corpus/'extra_motions'/Path(row['motion_path']).name;np.savez_compressed(path,fps=50.,**arrays);row.update(unprojected_motion_path=row['motion_path'],projection_teacher_run=r['run'],motion_path=str(path),motion_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),projection_accepted=True)
 records.append(row)
 with np.load(row['motion_path']) as z:
  for k in keys:assert np.isfinite(z[k]).all();buffers[k].append(z[k].astype(np.float32))
np.savez_compressed(corpus/'training_motions.npz',fps=50.,**{k:np.concatenate(v) for k,v in buffers.items()});(corpus/'manifest.json').write_text(json.dumps(records,indent=2));(corpus/'clips.json').write_text(json.dumps(dict(ends=np.cumsum([r['frames'] for r in records]).tolist(),records=records),indent=2));summary=dict(clips=len(records),new_source_count=307,accepted=sum(r['accepted'] for r in audit),physical_complete=sum(r['complete'] for r in audit),frames=sum(r['frames'] for r in records),per_class={t:dict(planned=sum(r['task']==t for r in audit),accepted=sum(r['task']==t and r['accepted'] for r in audit)) for t in sorted({r['task'] for r in audit})},rows=audit,scope=scope);(out/'projection_audit.json').write_text(json.dumps(summary,indent=2));(corpus/'audit.json').write_text(json.dumps(summary,indent=2));print('Accepted physical references',summary['accepted'],'of307',flush=True)
