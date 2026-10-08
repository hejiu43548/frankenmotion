"""Replay a source human motion through frozen SONIC, optionally calibrating bounded legs.

Input NPZ: poses_axisangle [T,66], joints_zup_m [T,24,3], native 20Hz.
Output: recorded free-base MuJoCo motion at 50Hz; not a hardware command file.
"""
import argparse,os,json,shutil
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--human-npz',type=Path,required=True)
    ap.add_argument('--skeleton',type=Path,required=True,help='Verified official_human_skeleton.npz')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--target-speed',type=float,help='Optional per-clip simulator calibration, G1 heading speed m/s')
    ap.add_argument('--sonic-root',type=Path,default=Path('/home/pku/frankenmotion/work/g1_sonic_official'))
    ap.add_argument('--turn-deg',type=float)
    ap.add_argument('--turn-start',type=float,default=0.)
    ap.add_argument('--turn-end',type=float,default=6.)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory; existing results are preserved')
    z=np.load(args.human_npz,allow_pickle=False)
    poses=z['poses_axisangle'];joints=z['joints_zup_m']
    if poses.ndim!=2 or poses.shape[1]!=66 or joints.shape!=(len(poses),24,3) or len(poses)<40:raise ValueError('Expected >=40 frames of native 20Hz pose66 and joint24')
    if not np.isfinite(poses).all() or not np.isfinite(joints).all():raise ValueError('Nonfinite source')
    if 'fps' in z and float(z['fps'])!=20.:raise ValueError('Resample source to 20Hz before migration')
    if args.target_speed is not None and not 0<args.target_speed<=1.:raise ValueError('Calibration target must be in (0,1] m/s')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['human','retarget','rollouts']:(out/name).mkdir()
    shutil.copy2(args.human_npz,out/'human/source.npz');shutil.copy2(args.skeleton,out/'official_human_skeleton.npz')
    os.environ['FULL_OUT']=os.environ['DIAG_OUT']=str(out);os.environ['SONIC_ROOT']=str(args.sonic_root.resolve())
    import full,g1_runtime as rt,smpl_direct as sm,smpl_bounded as sb
    m=rt.load_model();p=sm.Policy();full.ex.retarget(m)
    heading=None if args.turn_deg is None else dict(degrees=args.turn_deg,start=args.turn_start,end=args.turn_end)
    if heading is not None and not 0<=args.turn_start<args.turn_end<=(len(poses)-1)*.05+.051:raise ValueError('Heading interval must fit source duration')
    def run(*a,**kw):return sb.run(m,p,*a,**kw,heading=heading)
    rows=[run('source',1.,0.,'native')]
    if args.target_speed is not None:
        for a in [.7,1.,1.4,1.8,2.2,2.6]:
            for b in [-.08,0,.08]:
                if a==1. and b==0:continue
                rows.append(run('source',a,b,f'cal_a{a:.1f}_b{b:.2f}',cmd=args.target_speed))
        pool=[r for r in rows if r['complete']]
        best=min(pool,key=lambda r:abs(r['heading_speed_mps']-args.target_speed)) if pool else None
    else:best=rows[0] if rows[0]['complete'] else None
    full.save('calibration.json',rows)
    if best is None:
        full.save('result.json',dict(status='failed',reason='All candidates fell'));return
    selected=run('source',best['amplitude'],best['bias'],'selected',seed=5501,cmd=args.target_speed)
    validation=[]
    if args.target_speed is not None:
        for seed in [5502,5503]:validation.append(run('source',best['amplitude'],best['bias'],f'validate_{seed}',seed,args.target_speed))
    raw=np.load(out/'rollouts/selected.npz');states=raw['qpos']
    names=[m.joint(i).name for i in range(1,m.njnt)]
    np.savez_compressed(out/'g1_motion.npz',root_pos=states[:,:3],root_quat_wxyz=states[:,3:7],joint_pos=states[:,7:],joint_names=np.asarray(names),fps=50.,entry_frames=50,complete=selected['complete'])
    full.save('result.json',dict(status='complete' if selected['complete'] else 'failed',selected=selected,validation=validation,meets_turn_tolerance=None if heading is None else all(r.get('turn_success',False) for r in [selected]+validation),meets_speed_tolerance=None if args.target_speed is None else all(r.get('speed_success',False) for r in [selected]+validation),note='Simulation only. Source upper rotations and timing unchanged; bounded lower rotation edits when calibrated. This does not certify semantic fidelity or unseen-motion generalization. Coordinates Z-up, quaternions wxyz, explicit MJCF joint names; includes 1s entry.'))
    print('Saved',out/'g1_motion.npz')
if __name__=='__main__':main()
