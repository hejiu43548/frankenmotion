"""Parameterized generation and frozen retargeting, with one shared tracker for every task."""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path('/home/pku/frankenmotion');OLD=ROOT/'outputs_amass/franken_improve_20261003';OUT=ROOT/'outputs_amass/franken_unified_20261004';BASE=ROOT/'outputs_amass/franken_eleven_20261003';WORK=ROOT/'work'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--task',required=True);ap.add_argument('--command',required=True,type=float);ap.add_argument('--seed',type=int,default=93044000);ap.add_argument('--prompt-index',type=int,choices=range(4),default=0);ap.add_argument('--name',required=True);ap.add_argument('--stage',choices=['all','generate','retarget','simulate'],default='all');ap.add_argument('--development-checkpoint');ap.add_argument('--development-preview',action='store_true');a=ap.parse_args()
 if not a.name.replace('_','').replace('-','').isalnum():ap.error('Use letters, digits, underscore or hyphen for name')
 if a.development_checkpoint:
  assert not 94041000<=a.seed<94041400,'Reserved final-test seeds require frozen selection'
  checkpoint=Path(a.development_checkpoint).resolve();controller=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),preview=a.development_preview,task_routing=False,scope='Unfrozen development smoke request')
 else:
  frozen=OUT/'frozen_unified/protocol.json';f=json.loads(frozen.read_text());checkpoint=Path(f['checkpoint']);assert sha(checkpoint)==f['checkpoint_sha256'] and not f['task_routing']
  controller=dict(checkpoint=str(checkpoint),checkpoint_sha256=f['checkpoint_sha256'],preview=f['preview'],task_routing=False,scope='Frozen shared tracker',frozen_protocol_sha256=sha(frozen))
 meta_path=checkpoint.parent/'protocol.json'
 if a.development_checkpoint:
  if not meta_path.exists():meta_path=checkpoint.with_suffix('.json')
  meta=json.loads(meta_path.read_text()) if meta_path.exists() else {};controller['preview_offsets']=meta.get('preview_offsets',[5,10,20] if controller['preview'] else [])
 else:controller['preview_offsets']=f.get('preview_offsets',[5,10,20] if controller['preview'] else [])
 folder=OUT/'single_requests'/a.name
 if a.stage=='all':
  assert not folder.exists(),'Choose a fresh name; existing results are preserved'
  common=['--task',a.task,'--command',str(a.command),'--seed',str(a.seed),'--prompt-index',str(a.prompt_index),'--name',a.name]
  if a.development_checkpoint:common+=['--development-checkpoint',str(checkpoint)]+(['--development-preview'] if a.development_preview else [])
  for py,stage in [(ROOT/'.conda/bin/python','generate'),(WORK/'g1_sim_env/bin/python','retarget'),(WORK/'mjlab_stable_env/bin/python','simulate')]:
   subprocess.run([str(py),str(Path(__file__).resolve()),*common,'--stage',stage],cwd=ROOT,check=True)
  print('Result:',folder/'result.json');return
 if a.stage=='generate':
  sys.path.insert(0,str(WORK));import physical_adapter_20261003 as pa
  from core import torch,np,FK,sample,TASKS,RANGES
  from generate import inputs
  if a.task not in TASKS:ap.error('task must be one of '+','.join(TASKS))
  tid=TASKS.index(a.task);lo,hi=RANGES[tid]
  if not lo<=a.command<=hi:ap.error(f'validated command range is [{lo}, {hi}]')
  folder.mkdir(parents=True,exist_ok=False);(folder/'controller.json').write_text(json.dumps(controller,indent=2))
  ip=json.loads((OLD/'frozen_integrated_v3/protocol.json').read_text());weight=OLD/ip['generation'].get(a.task,ip['generation']['all_other_tasks']);assert sha(weight)==ip['weight_hashes'][str(weight)]
  src=next(r for r in json.loads((BASE/'evaluation_manifest.json').read_text()) if r['task']==a.task and r['source'].endswith(f'_p{a.prompt_index}_s0'))
  torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);model,_=pa.core.load_model(task_weights=weight);fk=FK('cuda');local,tx,cmd,controls=inputs(src,[a.command]);raw=sample(model,local,tx,tid,cmd,[a.seed],True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
  path=folder/'request.npz';np.savez_compressed(path,motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task=a.task,command=a.command)
  row=dict(task=a.task,source=f'{a.task}_p{a.prompt_index}_single',seed=a.seed,command=a.command,command_index=0,path=str(path));(folder/'manifest.json').write_text(json.dumps([row],indent=2));(folder/'generation.json').write_text(json.dumps(dict(weight=str(weight),sha256=sha(weight),prompt=src['prompt'],sampler='DDIM50 eta0 guidance1',output_editing=False),indent=2));return
 assert json.loads((folder/'controller.json').read_text())==controller
 row=json.loads((folder/'manifest.json').read_text())[0]
 assert row['task']==a.task and row['seed']==a.seed and row['command']==a.command
 sys.path.insert(0,str(WORK))
 if a.stage=='retarget':
  import unified_retarget_20261004 as rt
  references=folder/'references';references.mkdir(exist_ok=False);rt.init();assert rt.g.P is None
  reference=rt.work((row,str(references)));(folder/'reference_manifest.json').write_text(json.dumps([reference],indent=2));return
 simulation=folder/'simulation';assert not simulation.exists()
 env=dict(os.environ,UNIFIED_PREVIEW='1' if controller['preview'] else '0',BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT=str(simulation),BM_RESULTS=str(folder/'manifest.json'),BM_REFERENCE_DIR=str(folder/'references'),BM_CHECKPOINT=str(checkpoint),BM_QUIET_METRICS='1');env.pop('BM_WEIGHT_MAP',None);env.pop('BM_TASK',None)
 with (folder/'simulation.log').open('w') as log:subprocess.run([str(WORK/'mjlab_stable_env/bin/python'),str(WORK/'unified_probe_20261004.py')],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert 'overflow' not in (folder/'simulation.log').read_text().lower() and sha(checkpoint)==controller['checkpoint_sha256']
 import numpy as np
 sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
 import transfer as tr
 from unified_metrics_20261004 import native_metrics
 from audit_results import TASKS,TOLS
 results=json.loads((simulation/'results.json').read_text());assert len(results)==1;r=results[0];assert not r.get('error') and r['checkpoint']==str(checkpoint)
 contract=json.loads((simulation/'contract.json').read_text());assert contract.get('preview_offsets',[5,10,20] if controller['preview'] else [])==controller['preview_offsets'];assert contract['contact_capacity_nconmax']==256 and contract['constraint_capacity_njmax']==2048
 m=tr.rt.load_model();height=tr.robot_height(m);ref=np.load(folder/'references/request_uniform.npz')['reference_qpos'];states=np.load(simulation/'request_actual.npz')['qpos'];actual=None
 if r['actual'] is not None:
  assert r['termination_time'] is None and len(states)==r['target_steps'];ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;assert ts[-1]>=want[-1]-1e-7
  pos=tr.get_positions(m,states);pos=np.stack([np.interp(want,ts,x) for x in pos.reshape(len(pos),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(pos,a.task,height);assert abs(actual['quantity']-r['actual']['quantity'])<1e-8 and actual['event_pass']==r['actual']['event_pass']
 result=dict(row,controller=controller,human=native_metrics(row),g1=tr.measure(tr.get_positions(m,ref),a.task,height),actual=actual,physical_complete=actual is not None,joint_pass=bool(actual is not None and actual['event_pass'] and abs(actual['quantity']-a.command)<=TOLS[TASKS.index(a.task)]),termination_time=r['termination_time'],raw_state_audited=True)
 (folder/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));print(json.dumps(result,indent=2,default=lambda x:x.item()))
if __name__=='__main__':main()
