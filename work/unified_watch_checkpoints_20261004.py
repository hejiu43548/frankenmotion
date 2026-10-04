"""Evaluate completed immutable snapshots while their training process continues."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004'
p=argparse.ArgumentParser();p.add_argument('--train-name',required=True);p.add_argument('--pid',required=True,type=int);p.add_argument('--preview',action='store_true');p.add_argument('--indices',nargs='+',type=int,default=[1000,2000,3000,4000]);a=p.parse_args()
def alive():
    path=Path(f'/proc/{a.pid}/cmdline')
    return path.exists() and 'unified_train_20261004.py' in path.read_bytes().decode().replace('\0',' ') and a.train_name in path.read_bytes().decode()
for index in a.indices:
    checkpoint=U/'training'/a.train_name/f'model_{index}.pt';ready=checkpoint.with_suffix('.pt.ready.json')
    while not ready.exists():
        assert alive(),f'Training process ended without completed snapshot {index}'
        time.sleep(20)
    expected=json.loads(ready.read_text())['checkpoint_sha256'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    name=f'{a.train_name}_step{index}_validation';assert not (U/'evaluation'/name).exists()
    args=[str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/unified_evaluate_20261004.py'),'--checkpoint',str(checkpoint),'--name',name]+(['--preview'] if a.preview else [])
    subprocess.run(args,cwd=R,check=True)
    subprocess.run([str(R/'work/g1_sim_env/bin/python'),str(R/'work/unified_assess_20261004.py'),'--name',name],cwd=R,check=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    print('Audited immutable snapshot',index,flush=True)
