"""Third intervention: periodic lower-body reference, fixed 6s horizon, original arms."""
import os,json,argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import run
import g1_runtime as rt
OUT=Path(os.environ['DIAG_OUT'])
original_reference=run.reference
FITS={}

def design(t,f):
    return np.stack([np.ones_like(t)]+[fn(2*np.pi*f*k*t) for k in range(1,4) for fn in [np.sin,np.cos]],axis=1)

def fit(case):
    z=np.load(run.SOURCES/(case+'.npz'));q=z['q'];t=np.arange(len(q))/20
    # Fit locomotion only, excluding start/end 0.5s. Frequency identified without physics.
    mask=(t>=.5)&(t<=t[-1]-.5)
    candidates=[]
    for f in np.linspace(.55,1.5,192):
        X=design(t[mask],f);coef=np.linalg.lstsq(X,q[mask,:12],rcond=None)[0]
        err=np.mean((X@coef-q[mask,:12])**2)
        candidates.append((err,f,coef))
    err,f,coef=min(candidates,key=lambda x:x[0]);FITS[case]=(f,coef)
    run.write('periodic_fit/'+case+'.json',dict(frequency_hz=f,fit_rmse_rad=float(np.sqrt(err)),coefficients=coef.tolist()))
    return f,coef

def reference(m,case,cadence=1.,amplitude=1.):
    q,quat,root=original_reference(m,case,1.,1.)
    f,coef=FITS.get(case) or fit(case)
    t=np.maximum(0,np.arange(len(q))*.02-1.)
    legs=design(t*cadence,f)@coef
    # Scale around the learned gait mean, preserving crouch/stance offsets.
    legs=coef[0]+(legs-coef[0])*amplitude
    a=np.clip(np.arange(len(q))*.02,0,1);a=a*a*(3-2*a)
    q[:,:12]=rt.Q0[:12]+a[:,None]*(legs-rt.Q0[:12])
    q=np.clip(q,m.jnt_range[1:,0]+1e-4,m.jnt_range[1:,1]-1e-4)
    # Both source tasks request straight walking. Explicit zero desired heading.
    quat[:]=[1.,0,0,0]
    return q,quat,root

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['sweep','validate'],required=True);args=ap.parse_args()
    run.reference=reference;m=rt.load_model();p=rt.Policy()
    if args.stage=='sweep':
        rows=[];grid=[(c,a) for c in [.7,.9,1.1,1.3,1.5,1.7] for a in [.8,1.,1.2,1.4,1.6]]
        run.write('periodic_protocol.json',dict(grid=grid,commands=[.2,.3,.4,.5,.6],tolerance_mps=.05,heading_tolerance_deg=15,horizon_s=5.96,arms='unretimed source',heading='zero as requested',note='Explicit robot-space gait reference editing; frozen generator/tracker. Different gait reference, not faithful original whole-body tracking.'))
        for case in run.CASES:
            for c,a in grid:
                label=f'periodic_{case}_c{c:.1f}_a{a:.1f}'
                saved=OUT/'rollouts'/(label+'.json')
                r=json.loads(saved.read_text()) if saved.exists() else run.evaluate(m,p,case,c,a,label)
                rows.append(r);run.write('periodic_sweep.json',rows)
    else:
        rows=json.loads((OUT/'periodic_sweep.json').read_text());results=[]
        for case in run.CASES:
            candidates=[r for r in rows if r['case']==case and r['complete'] and r['max_heading_drift_deg']<15 and r['joint_speed_max']<35]
            for cmd in [.2,.3,.4,.5,.6]:
                best=min(candidates,key=lambda r:abs(r['speed_mps']-cmd)) if candidates else None
                for seed in [1301,2702,3813]:
                    if best:
                        label=f'periodic_val_{case}_v{cmd:.1f}_s{seed}'
                        r=run.evaluate(m,p,case,best['cadence'],best['amplitude'],label,seed)
                        r.update(command_mps=cmd,method='periodic_gait',calibration_label=best['label'],success=bool(r['complete'] and abs(r['speed_mps']-cmd)<=.05 and r['max_heading_drift_deg']<15))
                    else:r=dict(case=case,command_mps=cmd,seed=seed,method='periodic_gait',success=False,complete=False,reason='no admissible calibration candidate')
                    results.append(r);run.write('periodic_validation.json',results)
        run.write('periodic_status.json',dict(stage='complete',count=len(results)))
if __name__=='__main__':main()
