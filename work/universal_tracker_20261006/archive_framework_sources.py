"""Snapshot installed control-framework sources used by this environment; no publishing."""
import json,hashlib,tarfile,platform,sys,subprocess,importlib,datetime
from pathlib import Path
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');out=D/'release';records={};paths=[]
for name in ['mjlab','rsl_rl']:
 module=importlib.import_module(name);root=Path(module.__file__).resolve().parent;files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.toml','.json','.xml']);records[name]=dict(installed_root=str(root),files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files});paths += [(p,name+'/'+str(p.relative_to(root))) for p in files]
archive=out/'framework_sources.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p,name in paths:tar.add(p,arcname=name)
source=Path('/home/pku/frankenmotion/work/mjlab_v111');git={}
for label,args in [('head',['rev-parse','HEAD']),('status',['status','--short'])]:
 r=subprocess.run(['git','-C',str(source),*args],capture_output=True,text=True);git[label]=r.stdout.strip() if r.returncode==0 else None
import torch,mujoco
(out/'framework_inventory.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,python=sys.version,platform=platform.platform(),cuda=torch.version.cuda,torch=torch.__version__,mujoco=mujoco.__version__,installed_sources=records,mjlab_build_source_git=git,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),caution='Code/config snapshot only; robot meshes, generator backbones, CUDA packages and datasets remain external dependencies. Native nominal compiled scene is separately supplied.'),indent=2));print('Archived',len(paths),'framework source/config files')
