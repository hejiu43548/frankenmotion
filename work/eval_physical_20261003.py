import sys,json,os
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import gmr_probe_20261003 as g
from concurrent.futures import ProcessPoolExecutor
ROOT=g.OUT;label=os.environ.get('EVAL_LABEL','physical');g.OUT=ROOT/(label+'_development_eval')
g.tr.rt.EFF[[4,5,10,11,13,14]]=50.;g.tr.rt.ARM[[4,5,10,11,13,14]]=2*.003609725
if __name__=='__main__':
 rows=json.loads((ROOT/(label+'_development_manifest.json')).read_text());res=[]
 with ProcessPoolExecutor(4,initializer=g.init) as pool:
  for r in pool.map(g.run,[(r,v) for v in ['stock','uniform'] for r in rows]):res.append(r);print(len(res),r['task'],r['variant'],r.get('error',r.get('actual')),flush=True)
 (ROOT/(label+'_development_results.json')).write_text(json.dumps(res,indent=2,default=lambda x:x.item()))
