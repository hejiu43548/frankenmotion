import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/pku/frankenmotion/outputs_amass');D=R/'turn_demo_20261005';out=D/'report';out.mkdir(exist_ok=True);rows=[]
for scene in sorted((D/'final12').glob('scene_*')):
 run=scene/'selected';r=json.loads((run/'result.json').read_text());v=r['command_metrics'];walk=json.loads((run/'walk_jitter_metrics.json').read_text());away=json.loads((run/'away_jitter_metrics.json').read_text());rows.append(dict(index=r['scene']['index'],success=r['success'],physical_complete=r['physical_complete'],reach_command=v['command_human_equiv_m'],reach_actual=v['actual_hold_wrist_forward_human_equiv_m'],reach_error=v['command_error_m'],hold_contact_s=v['longest_hold_contact_s'],hand_lowering_m=v['hand_lowering_m'],turn_command=v['turn_command_deg'],turn_actual=v['actual_turn_deg'],turn_error=v['turn_error_deg'],away_command=v['away_command_robot_m'],away_actual=v['actual_away_displacement_m'],away_error=v['away_error_m'],endpoint_error=v['final_reference_endpoint_error_m'],walk_jitter=walk['torso_angular_velocity']['highpass_5hz_rms'],away_jitter=away['torso_angular_velocity']['highpass_5hz_rms']))
assert len(rows)==12
with (out/'all_cases.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
labels=['Old smooth tracker','Previous reach tracker','Repaired unified tracker'];names=['old_gait_actor_diagnostic','selected','turn_full_actor_diagnostic'];diagnostics=[]
for i in [0,5,11]:
 diagnostics.append(dict(scene=i,metrics={label:json.loads((R/f'reach_demo_20261005/final_paired16/scene_{i:03d}'/name/'walk_jitter_metrics.json').read_text()) for label,name in zip(labels,names)}))
angular=np.array([[x['metrics'][k]['torso_angular_velocity']['highpass_5hz_rms'] for k in labels] for x in diagnostics]);actions=np.array([[x['metrics'][k]['action']['highpass_5hz_rms'] for k in labels] for x in diagnostics]);means=angular.mean(0);summary=dict(cases=12,successes=sum(r['success'] for r in rows),physical_complete=sum(r['physical_complete'] for r in rows),reach_means={str(c):float(np.mean([r['reach_actual'] for r in rows if r['reach_command']==c])) for c in [.3,.5]},mean_reach_error=float(np.mean([r['reach_error'] for r in rows])),turn_mean_abs_error_deg=float(np.mean([r['turn_error'] for r in rows])),failed_cases=[r for r in rows if not r['success']],jitter_paired_mean=dict(zip(labels,means.tolist())),jitter_reduction_fraction=float(1-means[2]/means[1]),repaired_vs_old_ratio=float(means[2]/means[0]),final_walk_jitter_range=[min(r['walk_jitter'] for r in rows),max(r['walk_jitter'] for r in rows)],final_away_jitter_range=[min(r['away_jitter'] for r in rows),max(r['away_jitter'] for r in rows)],minimum_hold_contact_s=min(r['hold_contact_s'] for r in rows))
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'jitter_diagnostics.json').write_text(json.dumps(diagnostics,indent=2));(out/'all_cases.json').write_text(json.dumps(rows,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False});colors=['#718096','#e77b43','#159d95'];fig,axs=plt.subplots(1,2,figsize=(12,4.6))
for ax,values,title,unit in zip(axs,[angular,actions],['Torso angular velocity above 5 Hz','Actor output above 5 Hz'],['RMS (rad/s)','RMS (normalized actions)']):
 ax.bar(np.arange(3),values.mean(0),color=colors,width=.6)
 for row in values:ax.plot(np.arange(3),row,'o-',color='#34445a',alpha=.5,lw=1)
 ax.set_xticks(np.arange(3),['Old smooth','Previous reach','Repaired']);ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.15)
fig.suptitle('Same generated reference, scene and initialization; only tracker changes',fontsize=13);fig.tight_layout();fig.savefig(out/'jitter_comparison.png',dpi=180);fig.savefig(out/'jitter_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4.4))
for pair in range(6):
 rr=rows[2*pair:2*pair+2];axs[0].plot([r['reach_command'] for r in rr],[r['reach_actual'] for r in rr],'o-',color='#159d95',alpha=.6)
axs[0].plot([.28,.55],[.28,.55],'--',color='#718096');axs[0].set(xlabel='Generated reach command (human-equivalent m)',ylabel='Physical held wrist reach (m)',title='Paired reach commands')
for row in rows:axs[1].scatter(row['turn_command'],row['turn_actual'],c='#159d95' if row['success'] else '#e77b43',s=45)
axs[1].plot([80,145],[80,145],'--',color='#718096');axs[1].set(xlabel='Requested turn (deg)',ylabel='Physical turn (deg)',title='One under-turn retained in results')
axs[2].plot(range(12),[r['walk_jitter'] for r in rows],'o-',label='Approach');axs[2].plot(range(12),[r['away_jitter'] for r in rows],'o-',label='Walk away');axs[2].axhline(means[1],color='#e77b43',ls='--',label='Previous tracker, paired mean');axs[2].set(xlabel='Unseen test case',ylabel='Torso angular velocity >5Hz RMS (rad/s)',title='Smooth locomotion in both directions');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(out/'command_response.png',dpi=180);fig.savefig(out/'command_response.pdf');plt.close(fig)
text=f'''本次任务：定位前进抖动，制作走近桌子、放手、收回放下、转向、走开的物理demo。

已验证的原因：同一生成参考、同一场景、相同初始化，只换tracker，旧平稳/上版reach/修复版的躯干5Hz以上角速度RMS平均分别为{means[0]:.4f}/{means[1]:.4f}/{means[2]:.4f} rad/s。上版tracker微调造成的行走退化是直接证据支持的主要原因。具体哪项PPO奖励贡献最大未逐项消融，不能单独归因于某一个奖励，更不能笼统归因于SONIC或FrankenMotion。

修复：以旧平稳actor为初始化，训练期间以旧行走actor、reach actor及SONIC提供监督，蒸馏成一个361输入、29输出的固定actor。完整序列补入转向和再次起步。推理没有按动作选权重、动作混合或动作滤波。量化抖动下降{summary['jitter_reduction_fraction']*100:.1f}%，修复版为旧平稳版的{summary['repaired_vs_old_ratio']:.2f}倍，接近原有水平。

生成命令：walk为G1根位移距离与方向；reach为相对骨盆的手腕前伸距离，以human-equivalent米计，0.3/0.5m并非G1手掌的实际移动距离。reach、放手、收回放下来自冻结的FrankenMotion reach生成器。turn来自已有TaskControl/RootControl条件模型，原范围0.45–1.5rad；90/135度用四次同噪声采样、根据生成FK端点调整注入值，无采样后关节角/转角修改。180度有饱和，本次不宣称覆盖。

坐标与衔接：GMR将人体动作转换为G1动作；没有桌面目标驱动的手臂IK。保留片段之间的插值与最终名义站姿。开始转向和离开时，根据当时实际根位置/朝向对后续参考施加一次刚性XY/yaw变换，使相对命令有正确坐标基准；不修改生成关节角、关节速度、时间或物理状态。每次变换记录在anchor_events.json。初始参考initial_motion.npz和变换后motion.npz均保留。它是参考管理，不是tracker切换，不是视频后处理。

独立测试：冻结tracker后用6个新随机桌子布局、每个0.3/0.5配对，共12条。全部无摔倒，完整阶段通过{summary['successes']}/12；全部接触和收手完成。一个135度测试实际112.76度，严格按失败保留，不调宽阈值。0.3/0.5实际均值为{summary['reach_means']['0.3']:.3f}/{summary['reach_means']['0.5']:.3f}m；最短连续采样接触{summary['minimum_hold_contact_s']:.2f}s。最终动作视频选择scene000（90度、reach0.3）和scene003（135度、reach0.5），配对reach展示scene002/003；其余结果完整保留。

仿真与模型：原生MuJoCo3.5真实闭环物理积分，50Hz控制，视频25fps、1倍速，渲染未平滑姿态。仅初始化写入qpos/qvel。G1 29DoF机身，固定rubber-hand网格与胶囊碰撞体，并非三指Dex3灵巧手；本结果不包含手指控制或抓取。桌子尺寸和高度固定、位置与朝向随机且已知，无感知、真机或域随机化验证。接触按50Hz采样的力与桌面法向判断，不证明采样间连续接触。高频指标诊断抖动，不能替代完整自然度评价。

文件：all_cases.csv列出每一条；jitter_diagnostics.json保存3组严格对照；服务器保留全部生成参考、训练数据、模型和原始状态。新实验没有push。
'''
(out/'实验说明.txt').write_text(text);print(json.dumps(summary,indent=2))
