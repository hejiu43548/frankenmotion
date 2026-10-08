import os,json,hashlib,csv
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import full,g1_runtime as rt
OUT=full.OUT;DEL=OUT/'delivery';DEL.mkdir(exist_ok=True)
def load(n):return json.loads((OUT/(n+'.json')).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sets={n:load(n) for n in ['baseline_results','smpl_results','contact_results','contact_v2_results','bounded_results','bounded_validation','smpl_bounded_results','smpl_bounded_validation','heading_results']}
    summary={};rawerror=[]
    for name,rows in sets.items():
        info=dict(total=len(rows),complete=sum(r['complete'] for r in rows))
        if 'validation' in name:
            info.update(speed_success=sum(r['speed_success'] for r in rows),mae_mps=float(np.mean([abs(r['heading_speed_mps']-r['command_mps']) for r in rows])))
        elif name.startswith('contact'):
            info.update(speed_success_heading_metric=sum(r['complete'] and abs(r['heading_speed_mps']-r['command_mps'])<=.05 for r in rows),speed_and_fidelity_success=sum(r['complete'] and abs(r['heading_speed_mps']-r['command_mps'])<=.05 and r['fidelity_pass'] for r in rows))
        if name=='heading_results':info.update(turn_success=sum(r['turn_success'] for r in rows),max_turn_error_deg=max(abs(r['turn_deg']-r['requested_turn_deg']) for r in rows))
        summary[name]=info
        for r in rows:
            z=np.load(OUT/'rollouts'/(r['label']+'.npz'));s=z['qpos']
            assert np.isfinite(s).all() and np.isfinite(z['action']).all()
            if r['complete']:
                yaw=np.unwrap(Rotation.from_quat(s[:,[4,5,6,3]]).as_euler('xyz')[:,2]);head=(yaw[50:-1]+yaw[51:])/2;dv=np.diff(s[50:,:2],axis=0)
                speed=np.mean(dv[:,0]*np.cos(head)+dv[:,1]*np.sin(head))/.02
                rawerror.append(abs(speed-r['heading_speed_mps']))
    summary['raw_speed_recalculation_max_error']=max(rawerror)
    summary['main_rollout_count']=sum(len(a) for a in sets.values())
    summary['cli_combined']=load('cli_combined/result')
    summary['smpl_fk_parity']=load('smpl_fk_parity')
    summary['source_upper_rotation_delta_max_rad']=max(r['source_upper_delta_max_rad'] for r in sets['smpl_bounded_validation'])
    summary['source_leg_rotation_delta_max_rad']=max(r['source_leg_delta_max_rad'] for r in sets['smpl_bounded_validation'])
    summary['limitations']=['Six prompt families, one generator seed 9301, 18 parameter variants; speed calibration uses four variants only.','Three small initial-velocity perturbations are not three unseen motions. Per-source selection used the simulator.','All speed values are physical G1 heading speed; no human-equivalent scaling. Contact summaries recompute heading speed for consistency.','Source upper rotations unchanged does not imply perfect actual arm tracking. Bounded lower edits up to 0.35 rad change the original motion.','No generator or SONIC weight training, terrain tests, hardware, or full semantic/naturalness certification.','Turning correction replaces global yaw intent; stop timing has not been solved or certified.']
    (DEL/'summary.json').write_text(json.dumps(summary,indent=2))
    audit=load('source_transfer_audit');(DEL/'source_transfer_audit.json').write_text(json.dumps(audit,indent=2))
    rows=[]
    for method in ['bounded','smpl_bounded']:
        for r in sets[method+'_validation']:
            rows.append(dict(method=method,case=r['case'],command_mps=r['command_mps'],actual_heading_mps=r['heading_speed_mps'],seed=r['seed'],complete=r['complete'],speed_success=r['speed_success'],amplitude=r['amplitude'],bias=r['bias']))
    with (DEL/'speed_validation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    prov=dict(remote_output=str(OUT),sonic_commit='b042411fae38ee4d1af9aac82a37a1f8d14d6dd0',files={str(p):sha(p) for p in list((OUT/'code').glob('*.py'))+[rt.XML,rt.POLICY/'model_encoder.onnx',rt.POLICY/'model_decoder.onnx',OUT/'official_human_skeleton.npz']})
    (DEL/'provenance.json').write_text(json.dumps(prov,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,method in zip(axs,['bounded','smpl_bounded']):
        data=sets[method+'_validation']
        for case in sorted(set(r['case'] for r in data)):
            xs=[.25,.4,.55];ys=[np.mean([r['heading_speed_mps'] for r in data if r['case']==case and r['command_mps']==v]) for v in xs]
            ax.plot(xs,ys,'o-',label=case)
        x=np.linspace(.2,.6,40);ax.plot(x,x,'k--');ax.fill_between(x,x-.05,x+.05,color='gray',alpha=.12)
        ax.set(xlabel='Command (physical G1 m/s)',ylabel='Actual heading speed (m/s)',title=method+f": {sum(r['speed_success'] for r in data)}/36",ylim=(.1,.65));ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(DEL/'speed_response.png',dpi=160);plt.close(fig)
    m=rt.load_model();m.vis.global_.offwidth=400;m.vis.global_.offheight=400;d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=400,width=400)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.8;camera.azimuth=125;camera.elevation=-15
    font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    writer=imageio.get_writer(str(DEL/'source_preserving_transfer.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
    try:
        for case in ['01_walk_speed_v1','04_walk_wave_v3','04_walk_wave_v1']:
            labels=['baseline_'+case,'smpl_'+case,f'smpl_bounded_validate_{case}_v0.55_s4401']
            data=[np.load(OUT/'rollouts'/(label+'.npz'))['qpos'] for label in labels]
            speeds=[json.loads((OUT/'rollouts'/(label+'.json')).read_text())['heading_speed_mps'] for label in labels]
            for t in range(0,len(data[0]),2):
                canvas=Image.new('RGB',(1200,550),(16,23,34));draw=ImageDraw.Draw(canvas)
                draw.text((15,10),'Source-preserving transfer | actual free-base G1 simulation',font=font(24),fill='white')
                draw.text((15,43),case+' | upper source rotations and timeline retained | third panel command 0.55 m/s',font=font(17),fill='#c3d3e5')
                for k,states in enumerate(data):
                    d.qpos[:]=states[t];mujoco.mj_forward(m,d);camera.lookat[:]=[d.qpos[0],d.qpos[1],.65];renderer.update_scene(d,camera=camera);canvas.paste(Image.fromarray(renderer.render()),(k*400,80))
                    draw.text((k*400+12,486),['G1 direction IK + mode0','Native human SMPL + mode2','SMPL + bounded leg correction'][k],font=font(18),fill='white')
                    color='#65dfbc' if k<2 or abs(speeds[k]-.55)<=.05 else '#ff9b83'
                    draw.text((k*400+12,516),f'Actual {speeds[k]:.3f} m/s | t={(t+1)/50:.2f}s',font=font(18),fill=color)
                writer.append_data(np.asarray(canvas))
                if t==174:canvas.save(DEL/(case+'_preview.png'))
    finally:writer.close();renderer.close()
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
