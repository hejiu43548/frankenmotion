"""Frozen-input paired confirmation for two kinematic conversions, SONIC v1.1.
Run only after selecting and hashing the candidate manifest/checkpoints.
Does not tune any parameters or exclude failed planned requests.
"""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import gmr_probe_20261003 as g
from concurrent.futures import ProcessPoolExecutor
ROOT=g.OUT

def init():
 g.tr.rt.EFF[[4,5,10,11,13,14]]=50.;g.tr.rt.ARM[[4,5,10,11,13,14]]=2*.003609725;g.init()

def run(item):
 row,method,folder=item;folder=Path(folder);stem=Path(row['path']).stem;dest=folder/(stem+'_'+method+'.json')
 if dest.exists():return json.loads(dest.read_text())
 start=time.monotonic();r=dict(row,method=method);tr=g.tr;np=g.np
 try:
  z=np.load(row['path'])
  if method=='direction':
   q,quat,root,meta=tr.prepare_reference(g.M,z);states=np.c_[root,quat,q]
  elif method in ['stock','uniform']:
   states=g.convert(z,method);q=states[:,7:];quat=states[:,3:7];root=states[:,:3]
  else:raise ValueError(method)
  ref=tr.measure(tr.get_positions(g.M,states),row['task'],tr.robot_height(g.M));arrays,fall=tr.prior_run.rollout(g.M,g.P,*tr.upsample(q,quat,root));ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(q))*.05;actual=None
  if fall is None and ts[-1]>=want[-1]-1e-7:
   ps=tr.get_positions(g.M,arrays['qpos']);p=np.stack([np.interp(want,ts,x) for x in ps.reshape(len(ps),-1).T],1).reshape(len(q),24,3);actual=tr.measure(p,row['task'],tr.robot_height(g.M))
  r.update(g1=ref,actual=actual,fall_time=fall);np.savez_compressed(folder/(stem+'_'+method+'.npz'),reference_qpos=states,**arrays,time_s=ts)
 except Exception as exc:r['error']=repr(exc)
 r['wall_s']=time.monotonic()-start;tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(r,default=lambda x:x.item()));tmp.replace(dest);return r
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--name',required=True);ap.add_argument('--methods',default='direction,uniform');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();folder=ROOT/'confirmation'/a.name;folder.mkdir(parents=True,exist_ok=True)
 raw=Path(a.manifest).read_bytes();rows=json.loads(raw);methods=a.methods.split(',');protocol=dict(manifest=a.manifest,manifest_sha256=hashlib.sha256(raw).hexdigest(),n=len(rows),methods=methods,controller='SONIC v1.1 mode0',initialization='standing and 1s transition',effort_ankles_waist=50,armature_ankles_waist=2*.003609725,failures='preserve every request; missing metric charged E_all=1')
 (folder/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
 with ProcessPoolExecutor(a.workers,initializer=init) as pool:
  for r in pool.map(run,[(r,m,str(folder)) for m in methods for r in rows]):
   results.append(r)
   if len(results)%20==0:print('completed',len(results),'/',len(rows)*len(methods),flush=True)
 (folder/'results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
