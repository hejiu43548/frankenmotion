"""Build repository result tables from completed final audits, without selection."""
import csv,json
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';REPO=R/'work/git_publish_g1';read=lambda p:json.loads(p.read_text())
assert (U/'final_pipeline_complete.json').exists()
selected=read(U/'frozen_unified/protocol.json');labels={'unified_final':'统一权重','routed_baseline_final':'冻结的混合方案'};audits={n:read(U/'evaluation'/n/'audit.json') for n in labels};summaries={n:read(U/'evaluation'/n/'summary.json') for n in labels};challenge=read(U/'evaluation/unified_final_challenges/audit.json');export=read(U/'frozen_unified/actor.json')
assert all(a['aggregate']['requests']==880 and a['raw_max_error']<1e-8 and a['overflow_warnings']==0 for a in audits.values())
assert audits['unified_final']['single_checkpoint_sha256']==selected['checkpoint_sha256']==export['checkpoint_sha256']
text=['# 统一 G1 tracker：独立测试结果','', '本轮交付的是一套全动作共用的 tracker：同一观测构造、同一归一化、同一 MLP、同一权重。没有按动作切换 tracker，也没有把上下半身交给两个网络。实现采用 BeyondMimic 的全身策略接口进行多动作联合训练；没有直接平均 SONIC 与 BeyondMimic 的权重。','',f"最终权重由全部 11 类动作的开发集总体误差选择：`{selected['selected']}`。独立测试使用 880 条新噪声样本，在权重冻结后生成。",'', '| 方案 | 全程完成 | 动作语义通过 | 语义与参数精度同时通过 | 宏平均 E_all↓ |','|---|---:|---:|---:|---:|']
for name,label in labels.items():
 a=audits[name]['aggregate'];text.append(f"| {label} | {a['complete']}/880 | {a['event']}/880 | {a['joint']}/880 | {a['macro_semantic_E_all']:.4f} |")
paired=read(U/'statistics/final_paired_comparison.json');delta=paired['macro_mean_difference'][0];lo,hi=paired['macro_percentile_95'][0]
text+=['',f'统一减混合的 E_all 差值为 {delta:.4f}，配对 prompt/noise-block bootstrap 的 95% 区间为 [{lo:.4f}, {hi:.4f}]。重采样将同一提示编号/噪声编号对应的全部十一类动作和五档命令保留在同一个块内；该区间只反映本批输入变化，不代表跨训练种子的不确定性。']
text+=['','E_all 对物理失败或语义失败记 1；其余为绝对命令误差除以命令范围并截断至 1，各任务等权。全程站稳并不等于执行成功：例如没有真正起跳会被语义检查判失败。','', '![Command responses](figures/final_command_responses.png)','', '图中横轴为命令，纵轴为相应测量量；距离和速度换算为 human-equivalent 单位，角度保留弧度。曲线为完成样本的中位数，色带为 P10–P90，不是置信区间。失败样本保留在总体误差中；每个点标出完成/计划数量。','', '| 动作 | 统一：完成/语义/联合通过 | 统一 E_all↓ | 混合：完成/语义/联合通过 | 混合 E_all↓ |','|---|---:|---:|---:|---:|']
for task in summaries['unified_final']:
 x=summaries['unified_final'][task]['actual'];b=summaries['routed_baseline_final'][task]['actual'];text.append(f"| {task} | {x['measurable']}/{x['event_pass']}/{x['joint_pass']} | {x['semantic_E_all']:.4f} | {b['measurable']}/{b['event_pass']}/{b['joint_pass']} | {b['semantic_E_all']:.4f} |")
text+=['','每类动作 80 条。上游生成器与 retarget 仍使用冻结 V3 的任务适配方案，本轮只统一 tracker。混合方案与统一策略各自保留原生执行器和动力学配置，因此这是整个执行方案的比较，不能单独归因于网络架构。','', '## 连续与组合动作','', '| 测试 | 计划 | 全程完成 | 所有组件语义通过 | 所有组件联合通过 |','|---|---:|---:|---:|---:|']
for task,s in challenge['summary'].items():text.append(f"| {task} | {s['planned']} | {s['complete']} | {s['all_component_events']} | {s['all_component_joint_pass']} |")
text+=['','连续序列覆盖抬手→挥手→前倾、前进→转向→后退、踢腿→跳跃→前伸、侧移→击打→前进。组合参考覆盖走路/后退与右臂挥手。每条完整序列只载入一次权重，内部边界不重置机器人；全部组件通过才记整条通过。该组使用新噪声，但组合形式在增强训练中出现过，不代表未见组合泛化。','', '## 开发过程与失败实验','', '| 开发候选 | 完成 | 语义 | 联合通过 | E_all↓ |','|---|---:|---:|---:|---:|']
for c in selected['candidates']:
 s=c['aggregate'];text.append(f"| {c['name']} | {s['complete']}/110 | {s['event']} | {s['joint']} | {s['macro_semantic_E_all']:.4f} |")
text+=['','开发集上的多次比较参与了模型选择，不能当作独立测试。普通联合微调曾明显退化；后续加入真实片段终止、训练数据平衡、连续/组合参考、预览归一化校准，以及统一的关节与根部跟踪奖励。短/长预览的配对续训共享同一开发集选出的父权重、训练数据、奖励和随机种子；每种只有一次训练，尚无跨训练种子的统计验证。','', '开发集跳跃诊断显示，retarget 参考保留约 0.20–0.45 米的机器人根部上升和下蹲准备，但 V4 完成样本只执行出约 0.05–0.10 米。该证据定位的是这组跳跃的跟踪损失，不足以证明所有参考动力学可行，也不能说明所有动作都没有 retarget 问题。','', '## 可复查性与限制','',f"- 策略 SHA-256：`{selected['checkpoint_sha256']}`。",f"- 导出 actor 输入 {export['input_dimensions']} 维、输出 29 维；预览帧偏移 `{export['preview_offsets']}`，参考频率 50 Hz。",f"- 最终从保存状态重算指标的最大误差：统一 {audits['unified_final']['raw_max_error']:.3g}，混合 {audits['routed_baseline_final']['raw_max_error']:.3g}；未发现接触/约束缓冲区溢出。", '- 原始物理轨迹、训练数据和全部检查点保留在服务器 `outputs_amass/franken_unified_20261004`。交付包包含唯一权重、actor 导出、代码、来源清单、审计和响应图。', '- 训练/开发/最终噪声种子分离，但文本模板和命令范围重合；不声称任意文本或范围外命令泛化。', '- 同一物理 rollout 的指标可以精确重算；不同 GPU rollout 不保证逐位一致。兼容性检查中曾出现约 7e-5 的最大状态差，重复运行也有同量级差异，未改动评测门槛。', '- 模型仍需配套的机器人关节顺序、观测顺序、动作缩放和原生控制参数。未执行真机部署，仿真通过不等于硬件可部署。','']
out=REPO/'experiments/g1_tracking/unified/RESULTS.md';out.write_text('\n'.join(text))
fig=out.parent/'figures';fig.mkdir(exist_ok=True)
import shutil
shutil.copy2(U/'figures/final_command_responses.png',fig/'final_command_responses.png')
path=U/'figures/command_responses.csv'
with path.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['pipeline','task','command','planned','complete','median','p10','p90'])
 for name in labels:
  rows=read(U/'evaluation'/name/'audited_results.json')
  for task in summaries[name]:
   for command in sorted({r['command'] for r in rows if r['task']==task}):
    rs=[r for r in rows if r['task']==task and r['command']==command];v=[r['actual']['quantity'] for r in rs if r['actual'] is not None];q=np.quantile(v,[.5,.1,.9]).tolist() if v else ['','',''];w.writerow([name,task,command,len(rs),len(v),*q])
print(out)
