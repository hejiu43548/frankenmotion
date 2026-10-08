import os,json,hashlib,csv
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import run,g1_runtime as rt
OUT=Path(os.environ['DIAG_OUT']);DEL=OUT/'delivery';DEL.mkdir(exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    shared=json.loads((OUT/'shared_gait_results.json').read_text())
    transfer=json.loads((OUT/'transfer_results.json').read_text())
    periodic=json.loads((OUT/'periodic_validation.json').read_text())
    original=json.loads((OUT/'variant_baselines.json').read_text())
    model=rt.load_model();raw_errors=[];upper_errors=[];rows=[]
    run.SOURCES=OUT/'variant_baselines/retarget'
    for r in shared:
        z=np.load(OUT/'rollouts'/(r['label']+'.npz'));s=z['qpos'];ref=z['reference_qpos']
        speed=float((s[-1,0]-s[50,0])/((len(s)-51)/50))
        raw_errors.append(abs(speed-r['speed_mps']))
        native_q,_,_=run.reference(model,r['case'],1,1)
        upper_errors.append(float(np.max(np.abs(ref[:,19:]-native_q[:,12:]))))
        assert np.isfinite(s).all() and np.allclose(z['fps'],50.)
        rows.append(dict(case=r['case'],command_mps=r['command_mps'],seed=r['seed'],actual_mps=speed,
                         error_mps=speed-r['command_mps'],success=r['success'],complete=r['complete'],
                         heading_drift_deg=r['max_heading_drift_deg'],joint_rmse_rad=r['joint_rmse_rad'],
                         source_upper_joint_rmse_rad=float(np.sqrt(np.mean((s[:,19:]-ref[:,19:])**2))),
                         contact_body_speed_mps=r['contact_body_speed_mean'],rollout_sha256=sha(OUT/'rollouts'/(r['label']+'.npz'))))
    with (DEL/'shared_gait_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=dict(root_invariance=json.loads((OUT/'root_invariance.json').read_text()),
        shared_success=sum(r['success'] for r in rows),shared_total=len(rows),
        shared_mae_mps=float(np.mean([abs(r['error_mps']) for r in rows])),
        shared_max_error_mps=max(abs(r['error_mps']) for r in rows),
        shared_max_heading_drift_deg=max(r['heading_drift_deg'] for r in rows),
        shared_source_upper_joint_rmse_mean=float(np.mean([r['source_upper_joint_rmse_rad'] for r in rows])),
        periodic_same_source_success=sum(r['success'] for r in periodic),periodic_same_source_total=len(periodic),
        periodic_transfer_success=sum(r['success'] for r in transfer),periodic_transfer_total=len(transfer),
        heldout_variant_transfer_success=sum(r['success'] for r in transfer if r['source_heldout']),
        heldout_variant_transfer_total=sum(r['source_heldout'] for r in transfer),
        raw_metric_max_error=max(raw_errors),upper_reference_preservation_max_error=max(upper_errors),
        commands_mps=[.25,.35,.45,.55],
        limitations=['Two prompt families, three numeric source variants each, shared diffusion seed 9301; not six independent unseen sources.',
        'Per-source calibration first; shared gait is explicit replacement of generated legs.',
        'Small initial velocity perturbations only; no terrain/push/mass robustness.',
        'Success is speed within 0.05 m/s, heading drift under 15 degrees, complete no-fall rollout; not full naturalness or semantic certification.',
        'No stop/transition controller, hardware, generator training or SONIC finetuning. Original upper-body reference preserved, actual tracking remains imperfect.'])
    run.write('delivery/summary.json',summary)
    # Every raw rollout is accounted for; calibration and failures remain available.
    files=list((OUT/'rollouts').glob('*.npz'))
    prov=dict(rollout_count=len(files),remote_output=str(OUT),runtime='CPU ONNX + MuJoCo; 2ms physics / 50Hz control',
              source_repo_commit='b042411fae38ee4d1af9aac82a37a1f8d14d6dd0',
              files={str(p):sha(p) for p in list((OUT/'code').glob('*.py'))+[rt.XML,rt.POLICY/'model_encoder.onnx',rt.POLICY/'model_decoder.onnx',rt.POLICY/'observation_config.yaml']})
    run.write('delivery/provenance.json',prov)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for case in run.CASES:
        a=[r for r in periodic if r['case']==case]
        for axis,records,label in [(axs[0],a,case),(axs[1],[r for r in transfer if r['case']==case],case)]:
            xs=sorted(set(r['command_mps'] for r in records));ys=[np.mean([r['speed_mps'] for r in records if r['command_mps']==x]) for x in xs];axis.plot(xs,ys,'o-',label=label)
    for case in sorted(set(r['case'] for r in transfer)):
        a=[r for r in transfer if r['case']==case and r['source_heldout']];xs=sorted(set(r['command_mps'] for r in a))
        if a:axs[1].plot(xs,[np.mean([r['speed_mps'] for r in a if r['command_mps']==x]) for x in xs],'.--',alpha=.8,label=case)
    for case in sorted(set(r['case'] for r in rows)):
        a=[r for r in rows if r['case']==case];xs=sorted(set(r['command_mps'] for r in a))
        axs[2].plot(xs,[np.mean([r['actual_mps'] for r in a if r['command_mps']==x]) for x in xs],'o-',label=case)
    for ax,title in zip(axs,['Per-source periodic gait: 30/30','Same map, different source gait: 29/48','Shared G1 gait + original upper body: 48/48']):
        ax.plot([.15,.65],[.15,.65],'k--',alpha=.5);x=np.linspace(.15,.65,50);ax.fill_between(x,x-.05,x+.05,color='gray',alpha=.12)
        ax.set(xlabel='Requested speed (m/s)',ylabel='Actual fixed-heading speed (m/s)',title=title,xlim=(.15,.65),ylim=(0,.75));ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.savefig(DEL/'command_response.png',dpi=170);plt.close(fig)
    # Two fixed representative clips, three commands each, recorded physical states.
    model.vis.global_.offwidth=420;model.vis.global_.offheight=420
    renderer=mujoco.Renderer(model,height=420,width=420);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.9;camera.azimuth=130.;camera.elevation=-15.
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=lambda n:ImageFont.truetype(fontpath,n)
    writer=imageio.get_writer(str(DEL/'g1_command_response.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
    try:
        for case in ['01_walk_speed_v1','04_walk_wave_v3']:
            sample=[next(r for r in shared if r['case']==case and r['command_mps']==c and r['seed']==6613) for c in [.25,.45,.55]]
            trajectories=[np.load(OUT/'rollouts'/(r['label']+'.npz'))['qpos'] for r in sample]
            for t in range(0,min(map(len,trajectories)),2):
                canvas=Image.new('RGB',(1260,580),(16,23,34));draw=ImageDraw.Draw(canvas)
                draw.text((16,10),'G1 actual simulation | shared calibrated legs + FrankenMotion upper body',font=font(23),fill='white')
                draw.text((16,43),case+' | frozen SONIC | fixed 1s entry + 5.96s action',font=font(17),fill='#b9c9da')
                for k,(states,r) in enumerate(zip(trajectories,sample)):
                    data.qpos[:]=states[t];mujoco.mj_forward(model,data);camera.lookat[:]=[data.qpos[0],data.qpos[1],.65]
                    renderer.update_scene(data,camera=camera);canvas.paste(Image.fromarray(renderer.render()),(420*k,85))
                    draw.text((420*k+15,510),f"Command {r['command_mps']:.2f}  Actual {r['speed_mps']:.3f} m/s",font=font(20),fill='#64e2bc')
                    draw.text((420*k+15,540),f"t={(t+1)/50:.2f}s | drift {r['max_heading_drift_deg']:.1f} deg",font=font(16),fill='white')
                writer.append_data(np.array(canvas))
                if t==174:canvas.save(DEL/(case+'_preview.png'))
    finally:writer.close();renderer.close()
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
