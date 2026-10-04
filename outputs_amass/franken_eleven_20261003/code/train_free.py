"""Numeric-condition finetuning through DDIM, frozen source and tracker.
Training shares prompt templates with the benchmark; confirmation uses held-out seeds.
No post-generation editing and no confirmation-state simulator fitting.
"""
import argparse,random,time
from core import *
from generate import inputs

def evaluate(m,fk,sources,step):
    rows=[]
    with torch.no_grad():
        for src in sources:
            if not src['source'].endswith('_p0_s0'):continue
            tid=src['task_id'];lo,hi=RANGES[tid];cmds=[lo,(lo+hi)/2,hi]
            local,tx,cmd,controls=inputs(src,cmds)
            raw=sample(m,local,tx,tid,cmd,[880000+tid]*3,True,root_controls=controls)
            base=sample(m,local,tx,tid,cmd,[880000+tid]*3,False,root_controls=controls)
            p=fk(raw);p0=fk(base);q=quantity(p,tid,HH/fk.height)
            pose=(((p-p[:,:,:1])-(p0-p0[:,:,:1]))**2).mean().sqrt()
            for c,v in zip(cmds,q.tolist()):rows.append(dict(task=src['task'],command=c,quantity=v,error_span=abs(v-c)/(hi-lo),relative_joint_rmse_m=float(pose)))
    score=np.mean([r['error_span']+.2*r['relative_joint_rmse_m'] for r in rows]);save(f'free_validation/step_{step:05d}.json',rows)
    return float(score)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2200);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);torch.manual_seed(81003);random.seed(81003)
    m,_=load_model(task_weights=OUT/'best.pt');fk=FK('cuda');params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-5)
    allsources=json.loads((OUT/'evaluation_manifest.json').read_text());sources={i:[r for r in allsources if r['task_id']==i and r['source'].endswith('_s0')] for i in range(11)}
    best=float('inf');start=0;wall=time.monotonic()
    if args.resume:
        state=torch.load(OUT/'free_last.pt',map_location='cuda',weights_only=False);m.denoiser.load_adapter(state['adapter']);opt.load_state_dict(state['optimizer']);start=state['step'];best=torch.load(OUT/'free_best.pt',map_location='cpu',weights_only=False)['score'];random.seed(81003+start)
    save('free_training_protocol.json',dict(steps=args.steps,learning_rate=3e-5,ddim_training_steps=10,validation_steps=50,train_noise_seeds='100000 + training step; disjoint from development and confirmation',validation_noise_seeds='880000 + task index',confirmation_noise_seeds='26010300 + prompt_index*100 + noise_index',prompt_overlap='Four templates per task shared between training and test; evaluates held-out noise and numeric commands, NOT held-out text generalization',objective='command Huber error/span + relative pose fidelity + velocity excess + teacher-support foot velocity',frozen='original generator, old root adapter, Sonic',initialization='stage1 best selected by held-out denoising loss'))
    for step in range(start,args.steps+1):
        if step%220==0 or step==args.steps:
            m.eval();score=evaluate(m,fk,allsources,step)
            state=dict(adapter=m.denoiser.adapter_state(),optimizer=opt.state_dict(),step=step,score=score,tasks=TASKS,ranges=RANGES)
            torch.save(state,OUT/'free_last.pt')
            if score<best:best=score;torch.save(state,OUT/'free_best.pt')
            save('free_training_status.json',dict(state='completed' if step==args.steps else 'training',step=step,total=args.steps,score=score,best=best,elapsed_s=time.monotonic()-wall,max_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20))
            print('FREE_VALIDATION',step,score,best,flush=True)
            if step==args.steps:break
        tid=step%11;src=random.choice(sources[tid]);lo,hi=RANGES[tid];value=random.uniform(lo,hi)
        local,tx,cmd,controls=inputs(src,[value]);seeds=[100000+step]
        with torch.no_grad():teacher=sample(m,local,tx,tid,cmd,seeds,False,steps=10,root_controls=controls);p0=fk(teacher)
        m.denoiser.train();opt.zero_grad(set_to_none=True)
        raw=sample.__wrapped__(m,local,tx,tid,cmd,seeds,True,steps=10,root_controls=controls);p=fk(raw);q=quantity(p,tid,HH/fk.height)
        command=torch.nn.functional.smooth_l1_loss((q-cmd)/(hi-lo),torch.zeros_like(cmd))
        relative=(p-p[:,:,:1])-(p0-p0[:,:,:1]);weights=torch.ones(24,device='cuda')
        if tid in [0,1,2,3]:weights[[17,19,21,23]]=.15
        if tid==7:weights[[2,5,8,11]]=.15
        pose=(relative.square()*weights[None,None,:,None]).mean()
        v=(p[:,1:]-p[:,:-1])*20;v0=(p0[:,1:]-p0[:,:-1])*20
        velocity=(v.norm(dim=-1)-v0.norm(dim=-1)*1.5-.5).clamp_min(0).square().mean()
        feet0=p0[:,:, [7,8]];support=(feet0[:,1:,:,2]<feet0[:,:,:,2].amin(1,keepdim=True)+.05)&((feet0[:,1:]-feet0[:,:-1]).norm(dim=-1)*20<.15)
        feetv=v[:,:, [7,8],:2];contact=(feetv.square().sum(-1)*support).sum()/support.sum().clamp_min(1)
        loss=command+2*pose+.002*velocity+.01*contact
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        opt.step()
        if step%22==0:
            row=dict(step=step,task=TASKS[tid],loss=float(loss),command=float(command),pose=float(pose),quantity=float(q),target=value,elapsed_s=time.monotonic()-wall)
            with (OUT/'free_train_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
if __name__=='__main__':main()
