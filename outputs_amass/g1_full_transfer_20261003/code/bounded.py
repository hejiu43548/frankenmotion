"""Per-source bounded residual calibration. Preserves source time and upper body."""
import json
import numpy as np
import full
import g1_runtime as rt

CASES=['01_walk_speed_v1','01_walk_speed_v3','04_walk_wave_v1','04_walk_wave_v3']
def reference(m,case,amplitude,bias):
    z=np.load(full.SOURCES/(case+'.npz'));q=z['q'].copy()
    mean=q[:,:12].mean(axis=0)
    residual=(amplitude-1)*(q[:,:12]-mean)
    residual[:,[0,6]]+=bias;residual[:,[4,10]]-=bias
    q[:,:12]+=np.clip(residual,-.35,.35)
    q=np.clip(q,m.jnt_range[1:,0]+1e-4,m.jnt_range[1:,1]-1e-4)
    return full.upsample(q,z['quat'],z['root'])

def main():
    m=rt.load_model();p=rt.Policy();rows=[];validation=[]
    full.save('bounded_protocol.json',dict(cases=CASES,amplitudes=[.7,1.,1.3,1.6,2.],biases=[-.08,0,.08],commands=[.25,.4,.55],delta_limit_rad=.35,heldout_initial_state_seeds=[4401,4402,4403],note='Per-source simulator calibration, not a learned or universal adapter. No retiming, source-leg replacement, or upper-body edit.'))
    for case in CASES:
        candidates=[]
        for a in [.7,1.,1.3,1.6,2.]:
            for b in [-.08,0,.08]:
                label=f'bounded_{case}_a{a:.1f}_b{b:.2f}'
                r=full.score(m,p,case,label,reference(m,case,a,b),extra=dict(amplitude=a,bias=b))
                rows.append(r);candidates.append(r);full.save('bounded_results.json',rows)
        for cmd in [.25,.4,.55]:
            pool=[r for r in candidates if r['complete']]
            best=min(pool,key=lambda r:abs(r['heading_speed_mps']-cmd))
            for seed in [4401,4402,4403]:
                r=full.score(m,p,case,f'bounded_validate_{case}_v{cmd:.2f}_s{seed}',reference(m,case,best['amplitude'],best['bias']),seed,extra=dict(command_mps=cmd,amplitude=best['amplitude'],bias=best['bias'],calibration_label=best['label']))
                r['speed_success']=bool(r['complete'] and abs(r['heading_speed_mps']-cmd)<=.05)
                validation.append(r);full.save('bounded_validation.json',validation)
if __name__=='__main__':main()
