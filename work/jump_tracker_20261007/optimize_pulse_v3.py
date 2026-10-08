import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fast_pulse as e
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';out=D/'pulse_search_v3';out.mkdir(exist_ok=False);rows=json.loads((R/'outputs_amass/universal_tracker_20261006/joint_corpus_expanded_corrected/manifest.json').read_text());train=[r for r in rows if r['task']=='jump' and r['seed'] in [990708,92041100,92041200,92041300]];assert len(train)==20

def score(p):
 results=[e.execute(dict(r,pulse_parameters=p.tolist())) for r in train]
 return dict(parameters=p.tolist(),loss=float(np.mean([r.get('loss',200.) for r in results])+.02*np.mean(p[:9]**2)),results=results)

if __name__=='__main__':
 rng=np.random.default_rng(710700);mean=np.r_[json.loads((D/'pulse_search_v2/best.json').read_text())['parameters'],0.,.15,-.08];std=np.r_[np.full(9,.12),.04,.2,.3,np.full(3,.2)];lo=np.r_[np.full(9,-.8),-.12,-.5,-2.,np.full(3,-.8)];hi=-lo;best=dict(parameters=mean.tolist(),loss=float('inf'));history=[]
 (out/'protocol.json').write_text(json.dumps(dict(training_sources=train,population=64,generations=16,seed=710700,scope='Specialized shared pulse residual across jump commands, training-only optimization, base actor fixed, no changed actuator limits, not a unified neural tracker. Height objective at50Hz; final benchmark audit remains unchanged20Hz.'),indent=2))
 with ProcessPoolExecutor(4,initializer=e.initialize,initargs=(str(D/'backup/actor.pt'),str(D/'backup/policy.pt'),str(out),'standing',False)) as pool:
  for generation in range(16):
   pop=np.clip(mean+rng.normal(size=(64,15))*std,lo,hi);pop[0]=np.zeros(15);pop[1]=best['parameters'] if best else mean;results=list(pool.map(score,pop));results.sort(key=lambda x:x['loss']);elite=np.array([r['parameters'] for r in results[:10]]);mean=.2*mean+.8*elite.mean(0);std=np.maximum(.2*std+.8*elite.std(0),np.r_[np.full(9,.025),.008,.03,.08,np.full(3,.03)]);best=results[0] if best is None or results[0]['loss']<best['loss'] else best;history.append(dict(generation=generation,best=best,mean=mean.tolist(),std=std.tolist()));(out/'history.json').write_text(json.dumps(history,indent=2));(out/'best.json').write_text(json.dumps(best,indent=2));print(generation,best,flush=True)
