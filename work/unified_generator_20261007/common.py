import sys,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007'
sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import physical_adapter_20261003 as pa
from core import torch,np,FK,sample,encode_control,HH,TASKS,RANGES
SOURCES=json.loads((pa.BASE/'evaluation_manifest.json').read_text())
SOURCES=[r for r in SOURCES if r['source'].endswith('_s0')]
def teacher_path(task):return D/'backup'/('jump.pt' if task=='jump' else 'kick.pt' if task=='kick' else 'physical.pt')
def inputs(src,cs,device,fk):
 z=torch.load(Path(src['prompt']).with_suffix('.pt'),map_location=device,weights_only=False);b=len(cs);local=z['local'][None].expand(b,-1,-1);tx={k:v.repeat((b,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v for k,v in z['tx'].items()};cmd=torch.tensor(cs,dtype=torch.float32,device=device);controls=None
 if src['task'] in ['walk','back_walk','turn']:
  values=torch.zeros(b,src['frames'],2,device=device)
  if src['task']=='turn':values[...,1]=-cmd[:,None]/((src['frames']-1)/20)
  else:values[...,0]=cmd[:,None]*fk.height/HH
  controls=encode_control(values,torch.ones(b,src['frames'],dtype=torch.bool,device=device))
 return local,tx,cmd,controls
