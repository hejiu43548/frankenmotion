"""Single-request frozen integrated V3 pipeline. Uses one of four cached task prompt templates."""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path('/home/pku/frankenmotion');OUT=ROOT/'outputs_amass/franken_improve_20261003';BASE=ROOT/'outputs_amass/franken_eleven_20261003';WORK=ROOT/'work'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--task',required=True);ap.add_argument('--command',required=True,type=float);ap.add_argument('--seed',type=int,default=91023001);ap.add_argument('--prompt-index',type=int,choices=range(4),default=0);ap.add_argument('--name',required=True);ap.add_argument('--stage',choices=['all','generate','simulate'],default='all');a=ap.parse_args()
 if not a.name.replace('_','').replace('-','').isalnum():ap.error('name must contain only letters, digits, underscore or hyphen')
 folder=OUT/'single_requests'/a.name
 if a.stage=='all':
  if folder.exists():raise FileExistsError('Choose a fresh --name; existing results are preserved')
  args=[x for x in sys.argv[1:]]
  if '--stage' in args:i=args.index('--stage');del args[i:i+2]
  for py,stage in [(ROOT/'.conda/bin/python','generate'),(WORK/'g1_sim_env/bin/python','simulate')]:subprocess.run([str(py),str(Path(__file__).resolve()),*args,'--stage',stage],check=True,cwd=ROOT)
  print('Result:',folder/'result.json');return
 if a.stage=='generate':
  sys.path.insert(0,str(WORK));import physical_adapter_20261003 as pa
  from core import torch,np,FK,sample,TASKS,RANGES
  from generate import inputs
  if a.task not in TASKS:ap.error('task must be one of '+','.join(TASKS))
  tid=TASKS.index(a.task);lo,hi=RANGES[tid]
  if not lo<=a.command<=hi:ap.error(f'validated command range is [{lo}, {hi}]')
  folder.mkdir(parents=True,exist_ok=False)
  ip=json.loads((OUT/'frozen_integrated_v3/protocol.json').read_text());weight=OUT/ip['generation'].get(a.task,ip['generation']['all_other_tasks']);assert sha(weight)==ip['weight_hashes'][str(weight)]
  src=next(r for r in json.loads((BASE/'evaluation_manifest.json').read_text()) if r['task']==a.task and r['source'].endswith(f'_p{a.prompt_index}_s0'))
  torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);model,_=pa.core.load_model(task_weights=weight);fk=FK('cuda');local,tx,cmd,controls=inputs(src,[a.command]);raw=sample(model,local,tx,tid,cmd,[a.seed],True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
  path=folder/'request.npz';np.savez_compressed(path,motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task=a.task,command=a.command)
  row=dict(task=a.task,source=f'{a.task}_p{a.prompt_index}_single',seed=a.seed,command=a.command,command_index=0,path=str(path));(folder/'manifest.json').write_text(json.dumps([row],indent=2));(folder/'generation.json').write_text(json.dumps(dict(weight=str(weight),sha256=sha(weight),prompt=src['prompt'],sampler='DDIM50 eta0 guidance1',output_editing=False),indent=2));return
 sys.path.insert(0,str(WORK));import confirmation_eval_20261003 as ce
 protocol=json.loads((OUT/'frozen_controllers_v1/protocol.json').read_text());row=json.loads((folder/'manifest.json').read_text())[0];route=protocol['task_route'][row['task']]
 ce.init()
 if row['task'] in ['raise_hand','lean']:
  import importlib.util
  frozen=OUT/'frozen_retarget_v2';rp=json.loads((frozen/'protocol.json').read_text());script=frozen/'task_retarget_v2_20261003.py';assert sha(script)==rp['script_sha256']
  spec=importlib.util.spec_from_file_location('frozen_task_retarget_v2',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.OUT=folder;result=module.run((row,rp['weight']));result['retarget_protocol']=str(frozen/'protocol.json')
 elif route=='sonic':result=ce.run((row,'uniform',str(folder)))
 else:
  np=ce.g.np;states=ce.g.convert(np.load(row['path']),'uniform');np.savez_compressed(folder/'request_uniform.npz',reference_qpos=states)
  bmout='single_requests/'+a.name+'/beyondmimic';env=dict(os.environ,BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT=bmout,BM_RESULTS=str(folder/'manifest.json'),BM_REFERENCE_DIR=str(folder),BM_WEIGHT_MAP=str(OUT/'frozen_controllers_v1/weight_map.json'),BM_QUIET_METRICS='1')
  subprocess.run([str(WORK/'mjlab_stable_env/bin/python'),str(WORK/'mjlab_probe_capacity_20261003.py')],env=env,cwd=ROOT,check=True);result=json.loads((OUT/bmout/'results.json').read_text())[0]
 result.update(capacity_correction_protocol=str(OUT/'capacity_correction_protocol.json') if route=='beyondmimic' else None,integrated_protocol=str(OUT/'frozen_integrated_v3/protocol.json'),selected_controller=route,controller_protocol=str(OUT/'frozen_controllers_v1/protocol.json'));(folder/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));print(json.dumps(result,indent=2,default=lambda x:x.item()))
if __name__=='__main__':main()
