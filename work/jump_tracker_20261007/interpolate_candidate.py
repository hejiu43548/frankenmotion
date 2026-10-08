"""One predeclared midpoint in weight space; same topology, normalization and seed."""
import torch,json,hashlib,shutil,subprocess,sys
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';U=R/'outputs_amass/universal_tracker_20261006';py=sys.executable
pa=D/'residual_training/reference_residual_v3_wide/actor_14999.pt';pb=D/'residual_training/reference_residual_v4_broad/actor_14999.pt';out=D/'residual_training/reference_residual_v5_midpoint';out.mkdir(exist_ok=False)
a=torch.jit.load(str(pa));b=torch.jit.load(str(pb));sa=a.state_dict();sb=b.state_dict();assert sa.keys()==sb.keys()
with torch.no_grad():
 for k in sa:
  if k.startswith('head.'):sa[k].copy_((sa[k]+sb[k])*.5)
  else:assert torch.equal(sa[k],sb[k]),k
a.load_state_dict(sa);a.eval();path=out/'actor.pt';a.save(str(path));meta=json.load(open(pa.with_suffix('.json')));meta.update(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training=str(out),sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pa,pb]},interpolation='Single residual MLP with arithmetic midpoint of v3/v4 head parameters; base and buffers byte-identical; no runtime ensemble or task routing');path.with_suffix('.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,out/'interpolate_candidate.py')
for suffix,manifest in [('dev',D/'dev_jump_manifest.json'),('extended_dev',D/'extended_dev_jump_manifest.json'),('regression',U/'native_eleven_manifest.json'),('natural',U/'natural_test_motion/manifest.json')]:
 name='residual_v5_'+suffix;subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(path),'--actor',str(path),'--manifest',str(manifest),'--name',name,'--workers','4'],check=True)
 if suffix=='natural':subprocess.run([py,str(R/'work/universal_tracker_20261006/assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/name)],check=True)
 else:subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
subprocess.run([py,str(W/'evaluate_table_actor.py'),'--checkpoint',str(path),'--actor',str(path),'--name','residual_v5_table','--workers','3'],check=True)
