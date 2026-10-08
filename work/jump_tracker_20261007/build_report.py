"""Generate a Chinese plain-text report from audited outputs, not hand-entered scores."""
import argparse,json,csv,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();D=Path(a.root);E=D/'general_evaluation';O=D/'deliverables';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
summary=read(O/'final_summary.json');selection=read(D/'candidate_freeze.json');model=selection['source_model'];tag=selection['validation_tag'];final=summary['all'];b=final['baseline'];n=final['candidate']
def compact(s):return f"完整执行 {s['complete']}/{s['requests']}，跳跃事件 {s['event']}/{s['requests']}，高度+事件达标 {s['height_and_event_pass']}/{s['requests']}，完整样本高度 MAE {s['mae_completed_m']:.4f} m，含失败惩罚的 E_all {s['semantic_E_all']:.4f}"
lines=['G1 跳跃跟踪：三小时限时实验','',f"最终候选：{model}",f"权重 SHA256：{selection['checkpoint_sha256']}",'',
'结论',f"冻结后新生成的 60 条参考中，原 tracker {b['height_and_event_pass']}/60 条达到高度和跳跃事件要求，新 tracker {n['height_and_event_pass']}/60；完整执行从 {b['complete']}/60 变为 {n['complete']}/60。这次找到的是有明显改善的实验候选，还不能称为跳跃跟踪已经全面解决。",'',
'1. 这次改了什么',
'FrankenMotion 生成器、命令注入、GMR 重定向及测试参考都没有改。原有共享 tracker 的主干保持冻结，增加一个学习到的腿部残差头；主干和残差头一起存成一个 TorchScript 权重。所有动作运行同一份权重，没有 jump/走路/上半身的专家切换。',
'最终残差头读取 353 维参考特征：当前与 5 个未来时刻的关节位置、速度，加 5 个参考锚点位移长度。两层 256 维 ELU，输出髋 pitch、膝、踝 pitch 三个残差，左右腿对称使用，tanh 限幅到 ±0.8 rad。原主干仍使用 495 维参考和本体状态完成反馈稳定。输出加到原 29 维关节 PD 目标，不修改真实仿真中的根平移、不抬高力矩限额。控制频率 50 Hz，MuJoCo 200 Hz。',
'教师数据来自训练参考上的 MuJoCo 轨迹优化：用固定预算 CEM 搜索起跳、腾空和落地阶段的三个关节修正，再把成功轨迹蒸馏到残差头。最终候选使用 45 条跳跃教师、50 条其他动作教师，并用历史训练桌面状态和 307 条训练集动捕参考上的零残差约束保持其他能力。教师优化是离线训练步骤；最终新样本测试没有逐参考重新搜索。',
'这仍含跳跃专门构造的教师，不是“完全没有蒸馏”的通用训练方案，也不是从头训练出全新底层控制器。原有 BeyondMimic 系列主干及历史训练资产沿用。', '',
'2. 为什么旧 tracker 跳不好',
'固定的新测试首个随机源、0.55 m 命令下，重建的质心峰值向上速度由原 tracker 的约 1.45 m/s 增至新 tracker 的 2.26 m/s，参考约 2.30 m/s。这支持起跳冲量不足得到改善的判断；峰值速度本身不能代替完整跟踪评分。',
'GMR 后的骨盆高度、关节角和速度仍随命令变化，说明命令没有简单地在 retarget 时消失。旧 tracker 对高跳给出的起跳速度偏小，实际高度响应被压缩；只修改播放速度或增加简单竖直反馈没有解决。',
'参考还存在动力学不一致：在无接触的腾空区间，参考全身质心加速度没有严格满足重力抛物线。几何上看起来正确的参考，不等于每一帧都能被动力学忠实跟踪。旧逐帧奖励、起跳冲量和落地时序之间存在冲突，是当前证据支持的解释；不能把问题全部归因于某一个框架。',
'同样的模型和执行器限额下，逐参考物理优化能在 10 条开发参考上全部达标，说明至少这批运动存在更好的控制解。这个 10/10 是带仿真搜索的规划器结果，不能冒充零样本神经 tracker 的分数。', '',
'3. 冻结后的新噪声测试',
'先按开发集、历史验证集及回归结果选定一个全局权重并保存哈希，然后生成新随机种子样本；测试结果不再用于换权重或微调。4 个已用过的提示模板 × 2 个新噪声源 × 5 个标准高度 = 40 条；另有 20 条中间高度。不是新语言类别测试，也不是实机测试。',
'达标定义沿用旧实验：完整执行、通过跳跃/落地事件检查、human-equivalent 高度误差 ≤0.04 m。跌倒仍计入分母。高度是骨盆相对初始位置的最大上升量，按旧实验标尺 1.270119/1.048644 换算；因此 0.55 m 命令不能直接解释为机器人原生单位下离地 0.55 m。曲线的均值/P10–P90 仅用可测量完整轨迹，柱状成功率保留全部请求。']
for part,label in [('standard_grid','标准高度 0.25/0.325/0.40/0.475/0.55'),('interpolation','中间高度 0.275/0.35/0.425/0.50/0.525'),('all','合计')]:
 lines+=['',label,'原 tracker：'+compact(summary[part]['baseline']),'新 tracker：'+compact(summary[part]['candidate'])]
for name,label in [('final_baseline','原 tracker'),('final_candidate','新 tracker')]:
 p=E/name/'contact_audit.json'
 if p.exists():
  z=read(p);lines.append(f"接触几何独立复核，{label}：{z['complete_air_and_land']}/{z['requests']} 完整执行且峰值位于至少 60 ms 双脚离地段，并在之后恢复双脚接触。此检查不替换原始评分标准。")
lines+=['','新 tracker 的 3 次失败均在 0.55 m 档，终止时间为 3.22–3.46 秒，发生在落地阶段；这一档只有 2/8 达标，不能被整体 46/60 掩盖。', '', '4. 其他动作与桌面场景回归']
base=read(D/'baseline_metrics/projection_initial_eleven/eleven_summary.json');candidate=read(E/(tag+'_regression')/'eleven_summary.json')
for task in base:
 x=base[task]['actual'];y=candidate[task]['actual'];lines.append(f"{task}: 高度/速度/角度等任务各自的原达标定义 {x['joint_pass']}/{x['planned']} -> {y['joint_pass']}/{y['planned']}；完整 {x['measurable']} -> {y['measurable']}")
for path,label in [(D/'baseline_metrics/fresh_candidate_natural/fidelity_summary.json','原 tracker'),(E/(tag+'_natural')/'fidelity_summary.json','新 tracker')]:
 z=read(path)['results'];total=sum(r['requests'] for r in z.values());complete=sum(r['complete'] for r in z.values());strict=sum(r['accurate_complete'] for r in z.values());lines.append(f"54 条动捕回归，{label}：完整 {complete}/{total}，严格位置误差达标 {strict}/{total}。")
lines.append('动捕严格标准为完整执行且全局平均身体位置误差 ≤0.10 m、帧平均误差 P95 ≤0.25 m；这不等于动作语义或观感全部正确。上述回归集参与了候选选择，只能算开发回归。')
table=read(D/'table_evaluation'/(tag+'_table')/'summary.json');lines.append(f"历史桌面场景：{table['success']}/{table['planned']} 成功，{table['complete']}/{table['planned']} 完整执行。")
lines+=['','5. 延迟边界与组合方法',
'追加测试使用新样本中的 40 条标准高度参考。20 ms 是一个控制周期的动作目标延迟。预测补偿用当前完整仿真状态、已排队的 PD 目标和一个模型向前推演 20 ms，再让同一 actor 为预计到达的状态计算下一动作；物理环境没有被预测状态覆盖。它需要可靠状态估计和模型，不能直接等同于实机可用。']
for name,label in [('final_baseline_delay','原 tracker +20ms 延迟'),('final_baseline_predict','原 tracker +20ms 延迟+预测'),('final_candidate_delay','新 tracker +20ms 延迟'),('final_candidate_predict','新 tracker +20ms 延迟+预测'),('final_candidate_predict_mass','新 tracker +延迟+预测，真实仿真质量/惯量 +10%，预测模型仍为名义值')]:
 p=E/name/'eleven_audit.json'
 if p.exists():
  x=read(p)['aggregate'];lines.append(f"{label}：完整 {x['complete']}/{x['requests']}，高度+事件达标 {x['joint']}/{x['requests']}。")
lines+=['','6. 做过但没有选为最终结果的尝试',
'共享 PPO：增加跳跃采样、放松提前终止、竖直速度/高度/离地奖励；配套曝光量控制组；质心腾空抛物线修复后继续 PPO；放松腾空逐帧奖励并增加预测顶点奖励。这些是本次有限步数试验，失败不能解释为对应研究路线本身无效。',
'播放速度调整、简单竖直反馈、全局分阶段残差参数、逐参考 CEM、全状态残差头、只读参考的残差头、更多保持数据、权重插值、追加 40 条训练参考、加宽残差网络。完整评分见 experiment_catalog.csv；没有把开发集搜索的最佳成绩冒充最终测试。',
'不同候选存在取舍：较强跳跃拟合可能增加跌倒或损害其他动作。最终选择理由见 candidate_freeze.json；原稳定版本没有被替换。', '',
'7. 这次能说明什么、不能说明什么',
'能说明：在不改生成器、不改 GMR、不增加执行器能力的情况下，用物理可行的训练教师加共享残差，能显著改善这一范围内的跳跃高度响应。',
'还不能说明：所有跳跃形态都能稳定执行、全动作 generalist 已完成、延迟和接触扰动下足够稳健、或者可以直接部署真机。训练教师只覆盖当前参数化跳跃；预测补偿还依赖完整状态与模型。本次只有一个训练随机种子的主要比较，历史验证被反复使用，需依赖单独冻结测试来约束过拟合。',
'建议下一步把“起跳冲量—腾空—落地”作为统一动力学约束和课程，结合混合动作的 on-policy 训练、延迟/质量/摩擦随机化；保留本次原权重作为回退。当前实验不继续超时运行。', '',
'8. 文件与复现',
'最终候选：final_candidate/actor.pt 与 actor.json；原版备份：backup/；最终冻结依据：candidate_freeze.json。',
'新测试逐样本表：deliverables/paired_fresh_results.csv；曲线：deliverables/fresh_jump_comparison.png 和 .svg；固定首个提示/噪声源的 0.25、0.40、0.55 命令视频：visuals/fresh_fixed_comparison/fixed_source_comparison.mp4。视频不是挑最好看的种子；失败会保留标注。',
'远程实验根目录：/home/pku/frankenmotion/outputs_amass/jump_tracker_20261007；代码：/home/pku/frankenmotion/work/jump_tracker_20261007。关键脚本：train_residual_actor.py、residual_actor.py、plan_per_reference.py、evaluate_actor.py、evaluate_predictive.py、final_pipeline.py。现有环境 work/mjlab_stable_env/bin/python；生成器使用 .conda/bin/python。运行依赖保留在远程原项目的数据与模型资产，并非脱离项目即可运行的独立包。',
'本次未 push。无效运行、文件名冲突修复和元数据修正均留有记录；它们不计作物理实验失败，见 metadata_errata.json 和各 INVALIDATED.json。', '',
'调研来源（启发本次设计，不声称复现这些方法）：',
'BeyondMimic: https://arxiv.org/html/2508.08241v1',
'OmniTrack: https://arxiv.org/html/2602.23832v1',
'AdaMimic: https://arxiv.org/html/2510.14454v1']
(O/'实验报告_跳跃跟踪.txt').write_text('\n'.join(lines)+'\n')
catalog=[]
for p in sorted(E.glob('*/eleven_audit.json')):
 if (p.parent/'INVALIDATED.json').exists():continue
 x=read(p)['aggregate'];catalog.append(dict(run=p.parent.name,**x))
with (O/'experiment_catalog.csv').open('w',newline='') as f:
 fields=['run']+list(catalog[0].keys())[1:];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(catalog)
print(O/'实验报告_跳跃跟踪.txt')
