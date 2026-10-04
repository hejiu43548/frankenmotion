import argparse,time
from core import *
def inputs(src,commands,device='cuda'):
    z=torch.load(Path(src['prompt']).with_suffix('.pt'),map_location=device,weights_only=False);b=len(commands)
    local=z['local'][None].expand(b,-1,-1);tx={k:v.repeat((b,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v for k,v in z['tx'].items()}
    cmd=torch.tensor(commands,device=device,dtype=torch.float32)
    control=None
    if src['task'] in ['walk','back_walk','turn']:
        values=torch.zeros(b,src['frames'],2,device=device)
        if src['task']=='turn':values[...,1]=-cmd[:,None]/((src['frames']-1)/20)
        else:values[...,0]=cmd[:,None]*FK().height/HH
        control=encode_control(values,torch.ones(b,src['frames'],device=device,dtype=torch.bool))
    return local,tx,cmd,control
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',default='best.pt');ap.add_argument('--tag',default='task_adapter');ap.add_argument('--baseline',action='store_true');ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);m,_=load_model(task_weights=None if a.baseline else OUT/a.weights);fk=FK('cuda')
    sources=json.loads((OUT/'evaluation_manifest.json').read_text());folder=OUT/'generated'/a.tag;folder.mkdir(parents=True,exist_ok=True);rows=[];start=time.monotonic()
    for src in sources:
        if a.probe and not src['source'].endswith('_p0_s0'):continue
        commands=src['commands'];local,tx,cmd,controls=inputs(src,commands)
        paths=[folder/f"{src['source']}_c{i}.npz" for i in range(5)]
        if not all(p.exists() for p in paths):
            raw=sample(m,local,tx,src['task_id'],cmd,[src['seed']]*5,not a.baseline,root_controls=controls)
            with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);q=quantity(fk(raw),src['task'],HH/fk.height);net=quantity(fk(raw),'walk',HH/fk.height,True)
            for i,path in enumerate(paths):
                np.savez_compressed(path,motion=raw[i].cpu().numpy(),joints_zup_m=pos[i].cpu().numpy(),poses_axisangle=poses[i].cpu().numpy(),root_translation=root[i].cpu().numpy(),fps=20.,human_quantity=float(q[i]),net_forward_speed=float(net[i]),command=commands[i],human_height=fk.height,task=src['task'],source=src['source'])
        for i,path in enumerate(paths):
            z=np.load(path);rows.append(dict(task=src['task'],source=src['source'],seed=src['seed'],command=commands[i],command_index=i,path=str(path),human_quantity=float(z['human_quantity']),net_forward_speed=float(z['net_forward_speed'])))
        save(f'generated/{a.tag}_manifest.json',rows)
        print(src['source'],[round(r['human_quantity'],3) for r in rows[-5:]],'elapsed',round(time.monotonic()-start,1),flush=True)
    save(f'generated/{a.tag}_status.json',dict(state='completed',count=len(rows),weights=None if a.baseline else a.weights))
if __name__=='__main__':main()
