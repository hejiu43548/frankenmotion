import argparse, csv, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
ROOT = Path('/home/pku/frankenmotion')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
import src.prepare
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from control import RootControl, labels, root_signals, window_profile, encode_control

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)

def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)

def move(obj):
    if torch.is_tensor(obj): return obj.cuda(non_blocking=True)
    if isinstance(obj, dict): return {k: move(v) for k, v in obj.items()}
    if isinstance(obj, list): return [move(v) for v in obj]
    return obj

def load_model(config, ckpt):
    model = instantiate(config.diffusion)
    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    model.denoiser = RootControl(model.denoiser)
    model.cuda().eval()
    return model, {'epoch': checkpoint.get('epoch'), 'global_step': checkpoint.get('global_step')}

def prepare(model, batch):
    mask = batch['mask']
    x = model.motion_normalizer(batch['x'])*mask[..., None]*batch['stats_mask']
    y = {'mask': mask, 'length': batch['length'], 'tx': model.prepare_tx_emb(batch['tx'])}
    values, valid = labels(batch['x'], batch['length'])
    return x, y, values, valid

def objective(model, batch, training):
    x, y, targets, valid = prepare(model, batch)
    available = valid[..., None].expand_as(targets).clone()
    if training:
        # Independent omissions allow speed-only, yaw-only, and no-control use.
        available &= (torch.rand(len(x), 1, 2, device=x.device) > .15)
    y['root_control'] = encode_control(targets, available)
    t = torch.randint(model.timesteps, (len(x),), device=x.device)
    xt = model.q_sample(xstart=x, t=t, noise=torch.randn_like(x))*batch['mask'][..., None]
    xt = torch.cat((xt[..., :205], x[..., 205:]), -1)
    pred = model.denoiser(xt, y, t)
    diff = (pred[..., :205]-x[..., :205]).square()
    weights = torch.ones(205, device=x.device); weights[:4] = 5
    recon = (diff*weights*batch['mask'][..., None]).sum()/(batch['mask'].sum()*205)
    raw = model.motion_normalizer.inverse(pred)
    measured = window_profile(root_signals(raw), valid)
    speed_error = (measured[..., 0]-targets[..., 0])
    yaw_error = (measured[..., 1]-targets[..., 1])
    speed = (speed_error.square()*available[..., 0]).sum()/available[..., 0].sum().clamp_min(1)
    yaw = (yaw_error.square()*available[..., 1]).sum()/available[..., 1].sum().clamp_min(1)
    loss = recon + .1*speed + .1*yaw
    return loss, {'loss': loss.item(), 'reconstruction': recon.item(),
                  'speed_mae_mps': (speed_error.abs()*valid).sum().item()/valid.sum().item(),
                  'yaw_rate_mae_radps': (yaw_error.abs()*valid).sum().item()/valid.sum().item()}

@torch.no_grad()
def sample(model, batch, values, valid, random_seed=7301, steps=50):
    x, y, _, _ = prepare(model, batch)
    if values is not None: y['root_control'] = encode_control(values, valid)
    generator = torch.Generator(device='cuda').manual_seed(random_seed)
    xt = torch.randn(x.shape, generator=generator, device='cuda')
    timeline = np.linspace(model.timesteps-1, 0, steps, dtype=int).tolist()
    for index, step in enumerate(timeline):
        xt = torch.cat((xt[..., :205], x[..., 205:]), -1)*batch['mask'][..., None]
        t = torch.full((len(x),), step, device='cuda', dtype=torch.long)
        pred = model.denoiser(xt, y, t)
        if index == len(timeline)-1:
            xt = pred
        else:
            a = model.alphas_cumprod[step]
            b = model.alphas_cumprod[timeline[index+1]]
            eps = (xt-a.sqrt()*pred)/(1-a).sqrt()
            xt = b.sqrt()*pred+(1-b).sqrt()*eps
    return model.motion_normalizer.inverse(xt)[..., :205]

@torch.no_grad()
def generation_eval(model, dataset, out, tag):
    # Fixed held-out examples, fixed noise. Metrics concern root control, not overall realism.
    selected = np.linspace(0, len(dataset)-1, min(12, len(dataset)), dtype=int)
    records = []
    folder = out/'samples'/tag; folder.mkdir(parents=True, exist_ok=True)
    for idx in selected:
        batch = move(dataset.collate_fn([dataset[int(idx)]]))
        target, valid = labels(batch['x'], batch['length'])
        np.savez(folder/f'{idx}_reference.npz', motion=batch['x'][0,:,:205].cpu().numpy(), control=target[0].cpu().numpy(), valid=valid[0].cpu().numpy())
        for enabled in (False, True):
            raw = sample(model, batch, target if enabled else None, valid)
            measured = window_profile(root_signals(raw), valid)
            error = ((measured-target).abs()*valid[..., None]).sum((0,1))/valid.sum()
            n = int(batch['length'][0])
            name = f'{idx}_{"controlled" if enabled else "base"}'
            np.save(folder/(name+'.npy'), raw[0,:n].cpu().numpy())
            records.append({'index': int(idx), 'keyid': batch['keyid'][0], 'controlled': enabled,
                            'speed_mae_mps': error[0].item(), 'yaw_rate_mae_radps': error[1].item(),
                            'turn_error_deg': ((raw[0,:n-1,3]-batch['x'][0,:n-1,3]).sum()*180/np.pi).abs().item()})
        # Counterfactual targets: same text/noise, changed scalar control.
        # Moderate +/-20% speed and +/-0.15 rad/s yaw; not a realism guarantee.
        if int(idx) in set(selected[:4].tolist()):
            for factor, yaw_offset in ((.8,-.15),(1.2,.15)):
                requested=target.clone(); requested[...,0]*=factor; requested[...,1]+=yaw_offset
                raw=sample(model,batch,requested,valid)
                measured=window_profile(root_signals(raw),valid)
                error=((measured-requested).abs()*valid[...,None]).sum((0,1))/valid.sum()
                name=f'{idx}_speed{factor}_yaw{yaw_offset}'
                np.savez(folder/(name+'.npz'),motion=raw[0,:int(batch['length'][0])].cpu().numpy(),control=requested[0].cpu().numpy())
                records.append({'index':int(idx),'controlled':True,'counterfactual':True,'speed_factor':factor,'yaw_offset':yaw_offset,
                                'speed_mae_mps':error[0].item(),'yaw_rate_mae_radps':error[1].item()})
    atomic_json(folder/'metrics.json', records)
    return records

def audit(datasets, out):
    stats = {}
    for split, ds in datasets.items():
        entries = []
        with (out/f'labels_{split}.jsonl').open('w') as stream:
            for key in ds.keyids:
                ann = ds.annotations[key]
                raw = ds.motion_loader(ann['path'], ann['start'], ann['end'])['x']
                assert len(raw) >= 2 and torch.isfinite(raw).all(), key
                signals = root_signals(raw[:-1])
                row = {'keyid': key, 'path': ann['path'], 'start': ann['start'], 'end': ann['end'],
                       'frames': len(raw), 'mean_speed_mps': signals[:,0].mean().item(),
                       'mean_yaw_rate_radps': signals[:,1].mean().item(),
                       'signed_body_turn_deg': raw[:-1,3].sum().item()*180/np.pi}
                stream.write(json.dumps(row)+'\n'); entries.append(row)
        stats[split] = {'count': len(entries), 'speed_quantiles': np.quantile([r['mean_speed_mps'] for r in entries], [0,.25,.5,.75,.95,1]).tolist()}
    atomic_json(out/'data_audit.json', stats)
    print('DATA_AUDIT', json.dumps(stats), flush=True)

def run(args, out):
    seed(20260925); torch.set_num_threads(8)
    source = ROOT/'outputs_amass/official_20260916'
    ckpt = source/'logs/checkpoints/last.ckpt'
    cfg = OmegaConf.load(source/'config.json')
    cfg.data.preload = False
    cfg.data.motion_loader.cache = True
    cfg.data.text_encoder.rand_mask = False
    cfg.data.drop_cond = 0.0  # Avoid in-place mutation of cached captions in upstream dataset.
    config_path = out/'base_config.yaml'
    config_path.write_text(OmegaConf.to_yaml(cfg))
    model, origin = load_model(cfg, ckpt)
    provenance = dict(source_checkpoint=str(ckpt), source_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                      source_training=origin, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      args=vars(args), controls=['speed_mps', 'body_yaw_rate_radps'], fps=20, label_window_frames=20,
                      note='Frozen user-trained backbone. Adapter-only checkpoint requires source checkpoint and base_config.yaml.')
    atomic_json(out/'provenance.json', provenance)
    datasets = {s: instantiate(cfg.data, split=s) for s in ('train','val')}
    for ds in datasets.values(): ds.is_training = False
    # Pure extraction does not use validation statistics for normalization.
    if not args.resume: audit(datasets, out)
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=datasets['train'].collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=datasets['val'].collate_fn)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    best = float('inf'); stale = 0; start = 0; global_step = 0
    if args.resume:
        saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        assert saved['source_sha256'] == provenance['source_sha256']
        model.denoiser.load_adapter(saved['adapter']); optimizer.load_state_dict(saved['optimizer'])
        best=saved['best']; stale=saved['stale']; start=saved['epoch']; global_step=saved['global_step']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    # Meaningful initialization checks: controls initially preserve backbone output;
    # physical units checked independently on constant synthetic motion.
    if not args.resume:
        b = move(next(iter(val_loader)))
        x,y,v,m=prepare(model,b); t=torch.full((len(x),),99,device='cuda',dtype=torch.long)
        with torch.no_grad():
            original=model.denoiser(x,y,t)
            y['root_control']=encode_control(v,m)
            controlled=model.denoiser(x,y,t)
        assert torch.equal(original,controlled), 'zero-init changed baseline'
        toy=torch.zeros(1,41,205,device='cuda'); toy[...,1]=.1; toy[...,3]=np.pi/80
        vtest,mtest=labels(toy,torch.tensor([41],device='cuda'))
        assert torch.allclose(vtest[0,:40,0],torch.full((40,),2.,device='cuda'))
        assert abs(toy[0,:40,3].sum().item()-np.pi/2)<1e-5
        atomic_json(out/'startup_checks.json', {'zero_init_exact':True,'units_2mps_90deg_in_2s':True})
        if not args.smoke:
            generation_eval(model,datasets['val'],out,'before')
    for epoch in range(start,args.epochs):
        tick=time.time(); model.denoiser.train(); optimizer.zero_grad(); running=[]
        limit=min(len(train_loader),2) if args.smoke else len(train_loader)
        for i,batch in enumerate(train_loader):
            if i>=limit: break
            b=move(batch)
            loss,metrics=objective(model,b,True)
            assert torch.isfinite(loss), metrics
            # Correct scaling for the final short accumulation group.
            group_size=min(args.accumulate,limit-(i//args.accumulate)*args.accumulate)
            (loss/group_size).backward()
            if (i+1)%args.accumulate==0 or i+1==limit:
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(); global_step+=1
            running.append(metrics['loss'])
            if i%25==0:
                state=dict(state='training',epoch=epoch+1,max_epochs=args.epochs,batch=i+1,batches=limit,
                           global_step=global_step,loss=metrics['loss'],updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid())
                atomic_json(out/'status.json',state); print(json.dumps(state),flush=True)
        model.denoiser.eval(); values=[]
        # Fixed validation noise so early stopping is not driven by random t/noise.
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            torch.manual_seed(8101); torch.cuda.manual_seed_all(8101)
            for i,batch in enumerate(val_loader):
                if args.smoke and i>=2: break
                b=move(batch); _,metrics=objective(model,b,False); values.append((len(b['x']),metrics))
        n=sum(count for count,_ in values)
        val={key:sum(count*v[key] for count,v in values)/n for key in values[0][1]}
        improved=val['loss'] < best-1e-5
        if improved: best=val['loss']; stale=0
        else: stale+=1
        row={'epoch':epoch+1,'global_step':global_step,'train_loss':float(np.mean(running)),
             **{'val_'+k:v for k,v in val.items()},'epoch_seconds':time.time()-tick,'best':best}
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print('EPOCH',json.dumps(row),flush=True)
        checkpoint=dict(adapter=model.denoiser.adapter_state(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                        global_step=global_step,best=best,stale=stale,source_sha256=provenance['source_sha256'],
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate())
        torch.save(checkpoint,out/'last.tmp'); (out/'last.tmp').replace(out/'last.pt')
        if improved:
            torch.save(checkpoint,out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')
        if args.smoke:
            assert any(torch.count_nonzero(branch[-1].weight).item()>0 for branch in model.denoiser.residuals)
            assert all(p.grad is None for p in model.denoiser.base.parameters())
            with torch.no_grad():
                tx,ty,tv,tm=prepare(model,b); tt=torch.zeros(len(tx),device='cuda',dtype=torch.long)
                plain=model.denoiser(tx,ty,tt)
                ty['root_control']=torch.zeros(len(tx),tx.shape[1],4,device='cuda')
                assert torch.equal(plain,model.denoiser(tx,ty,tt)), 'missing control changed backbone'
                ty['root_control']=encode_control(tv,tm)
                assert not torch.equal(plain,model.denoiser(tx,ty,tt)), 'adapter has no effect'
            fresh,_=load_model(cfg,ckpt); fresh.denoiser.load_adapter(torch.load(out/'last.pt',weights_only=False)['adapter'])
            fresh.eval()
            bx,by,bv,bm=prepare(model,b); by['root_control']=encode_control(bv,bm)
            with torch.no_grad():
                assert torch.allclose(model.denoiser(bx,by,torch.zeros(len(bx),device='cuda',dtype=torch.long)),fresh.denoiser(bx,by,torch.zeros(len(bx),device='cuda',dtype=torch.long)),atol=1e-6)
            del fresh
            # Exercise actual multi-step inference and ensure finite generated motion.
            raw=sample(model,b,bv,bm,steps=5); assert torch.isfinite(raw).all()
            break
        if stale>=args.patience: break
    if not args.smoke:
        model.denoiser.load_adapter(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['adapter'])
        model.denoiser.eval(); generation_eval(model,datasets['val'],out,'after_best')
    atomic_json(out/'status.json',dict(state='completed',reason='smoke_passed' if args.smoke else ('early_stopping' if stale>=args.patience else 'max_epochs'),
                 epoch=epoch+1,global_step=global_step,best_validation_loss=best,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    (out/'DONE').write_text('Training and final evaluation completed successfully.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--patience',type=int,default=15); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume: raise RuntimeError('Existing checkpoint: use --resume or a fresh directory')
    if (out/'DONE').exists(): (out/'DONE').unlink()
    atomic_json(out/'status.json',dict(state='initializing',pid=os.getpid(),started=time.strftime('%Y-%m-%d %H:%M:%S')))
    try: run(args,out)
    except BaseException as exc:
        atomic_json(out/'status.json',dict(state='failed',error=repr(exc),time=time.strftime('%Y-%m-%d %H:%M:%S')))
        traceback.print_exc(); sys.exit(1)
