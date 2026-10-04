import argparse,time,random
from core import *
from control import labels
def batch(file,device='cuda'):
    z=torch.load(file,map_location=device,weights_only=False)
    return z['x'][None],z['stats_mask'][None],{k:v.to(device) if torch.is_tensor(v) else v for k,v in z['tx'].items()},z['quantity'].reshape(1)

def objective(m,fk,record,validation=False):
    raw,mask,tx,cmd=batch(record['path']);task=record['task_id'];x=m.motion_normalizer(raw)*mask
    controls=None
    if TASKS[task] in ['walk','back_walk','turn']:
        values,valid=labels(raw,torch.tensor([raw.shape[1]],device=raw.device));controls=encode_control(values,valid)
    y=make_y(m,x,tx,task,cmd,True,controls)
    if validation:
        gen=torch.Generator(device='cuda').manual_seed(91000+int(record['key']));noise=torch.randn(x.shape,generator=gen,device='cuda');t=torch.tensor([50],device='cuda')
    else:noise=torch.randn_like(x);t=torch.randint(0,m.timesteps,(1,),device='cuda')
    xt=m.q_sample(xstart=x,t=t,noise=noise);xt=torch.cat([xt[...,:205],x[...,205:]],-1)
    pred=m.denoiser(xt,y,t);diff=(pred[...,:205]-x[...,:205]).square();weights=torch.ones(205,device='cuda');weights[:4]=5
    recon=(diff*weights).mean();decoded=m.motion_normalizer.inverse(pred)[...,:205]
    q=quantity(fk(decoded),task,HH/fk.height)
    lo,hi=RANGES[task];control=torch.nn.functional.smooth_l1_loss((q-cmd)/(hi-lo),torch.zeros_like(cmd))
    loss=recon+.2*control
    return loss,dict(loss=float(loss.detach()),recon=float(recon.detach()),quantity_mae=float((q-cmd).abs().detach()),control=float(control.detach()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3300);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2601003);np.random.seed(2601003);random.seed(2601003)
    # Do not evict unrelated GPU services; this model fits within the available allocation.
    torch.cuda.set_per_process_memory_fraction(.15)
    m,cfg=load_model();fk=FK('cuda');params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4)
    records=json.loads((OUT/'data_manifest.json').read_text());train={i:[r for r in records if r['split']=='train' and r['task_id']==i] for i in range(11)}
    val=[r for r in records if r['split']=='val'];assert all(train.values()) and val
    start=0;best=float('inf')
    if args.resume:
        state=torch.load(OUT/'last.pt',map_location='cuda',weights_only=False);m.denoiser.load_adapter(state['adapter']);opt.load_state_dict(state['optimizer']);start=state['step'];best=state['best']
    save('training_protocol.json',dict(steps=args.steps,learning_rate=1e-4,trainable_parameters=sum(p.numel() for p in params),frozen='FrankenMotion base and existing root-control adapter',loss='weighted normalized motion reconstruction + 0.2 Huber(task error / command span), task quantity measured through native FK',seed=2601003,checkpoint_selection='held-out fixed-noise denoising loss, not test commands',root_controls='existing numeric speed/yaw branch active for walk/back_walk/turn only',task_order=TASKS))
    wall=time.monotonic()
    for step in range(start,args.steps+1):
        if step==0 or step%330==0 or step==args.steps:
            m.eval();metrics=[]
            with torch.no_grad():
                for r in val:
                    _,info=objective(m,fk,r,True);info['task']=r['task'];metrics.append(info)
            score=float(np.mean([r['loss'] for r in metrics]));state=dict(adapter=m.denoiser.adapter_state(),optimizer=opt.state_dict(),step=step,best=min(best,score),tasks=TASKS,ranges=RANGES)
            torch.save(state,OUT/'last.pt')
            if score<best:best=score;torch.save(state,OUT/'best.pt')
            save(f'validation/step_{step:05d}.json',metrics);save('training_status.json',dict(state='completed' if step==args.steps else 'training',step=step,total=args.steps,validation_loss=score,best=best,elapsed_s=time.monotonic()-wall,max_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20))
            print('VALIDATION',step,score,'best',best,flush=True)
            if step==args.steps:break
        m.denoiser.train();task=step%11;record=random.choice(train[task]);opt.zero_grad(set_to_none=True)
        loss,info=objective(m,fk,record)
        if not torch.isfinite(loss):raise FloatingPointError(info)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        opt.step()
        if step%55==0:
            info.update(step=step,task=TASKS[task],elapsed_s=time.monotonic()-wall)
            with (OUT/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
            print(json.dumps(info),flush=True)
if __name__=='__main__':main()
