"""Generate with the learned task command inside DDIM; no output editing."""
import argparse
from core import *
from generate import inputs
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--task',choices=TASKS,required=True);ap.add_argument('--command',type=float,required=True);ap.add_argument('--prompt-index',type=int,choices=range(4),default=0);ap.add_argument('--seed',type=int,default=710031);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--weights',type=Path,default=OUT/'task_adapter_deploy.pt');a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    tid=TASKS.index(a.task);lo,hi=RANGES[tid]
    if not lo<=a.command<=hi:raise ValueError(f'Validated command range is {lo}..{hi} {UNITS[tid]}')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);m,_=load_model(task_weights=a.weights);fk=FK('cuda')
    sources=json.loads((OUT/'evaluation_manifest.json').read_text());src=next(r for r in sources if r['source']==f'{a.task}_p{a.prompt_index}_s0')
    local,tx,cmd,controls=inputs(src,[a.command]);raw=sample(m,local,tx,tid,cmd,[a.seed],True,root_controls=controls)
    with torch.no_grad():p,poses,root=fk(raw,canonical=False,return_pose=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,motion=raw[0].cpu().numpy(),joints_zup_m=p[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),fps=20.,task=a.task,command=a.command,human_height=fk.height,seed=a.seed)
    print(a.output)
if __name__=='__main__':main()
