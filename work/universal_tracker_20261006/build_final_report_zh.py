"""Build the Chinese report only from collected frozen-test evidence."""
from pathlib import Path
import json,datetime
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';OUT=ROOT/'output/pdf/universal_tracker_20261006';OUT.mkdir(parents=True,exist_ok=True);data=json.loads((D/'report_data.json').read_text());frozen=data['frozen'];actor=data['actor'];controllers=data['controllers'];candidate=controllers['candidate'];broad=controllers['broad'];stable=controllers['stable'];N=candidate['aggregate']['requests'];assert N==880 and not frozen['task_routing']
pdfmetrics.registerFont(TTFont('Chinese','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
styles={k:ParagraphStyle(k,fontName='Chinese',fontSize=size,leading=leading,spaceAfter=space,textColor=colors.HexColor(color)) for k,size,leading,space,color in [('title',22,30,17,'#173c54'),('heading',15,22,11,'#176480'),('body',10,16.5,9,'#203342'),('small',8.3,12,7,'#52616c'),('cell',8.1,12,0,'#203342')]};story=[]
def p(text,kind='body'):story.append(Paragraph(escape(str(text)).replace('\n','<br/>'),styles[kind]))
def heading(text):p(text,'heading')
def page():story.append(PageBreak())
def table(rows,widths):
 t=Table([[Paragraph(escape(str(x)),styles['cell']) for x in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7092a6')),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d8e1e8')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,10)])
def figure(path,width=495):
 from PIL import Image as PIL
 path=Path(path);im=PIL.open(path);story.append(Image(str(path),width=width,height=width*im.height/im.width));story.append(Spacer(1,8))
def fraction(n,d):return f'{n}/{d} ({100*n/d:.1f}%)'
def natural_totals(c):
 r=c['natural']['results'];return {k:sum(v[k] for v in r.values()) for k in ['requests','complete','accurate_complete']}
cn=natural_totals(candidate);table_result=candidate['table'];agg=candidate['aggregate'];ba=broad['aggregate'];stats=data['statistics']['broad'];delta=stats['macro_difference'];ci=stats['macro_common_block_bootstrap95'];dim=actor['input_dimensions'];selected=frozen['selected']
p('FrankenMotion → G1\n统一 tracker 联合训练与扩展实验','title')
p('冻结测试报告 | 2026-10-06','heading')
p(f'最终选用 {selected}：一个共享控制器完成全部类别与场景的评估，无按任务切换权重。新噪声测试共 {N} 条，事件与数值精度联合通过 {fraction(agg["joint"],N)}；旧广覆盖共享版为 {fraction(ba["joint"],N)}。')
p(f'按每类“物理完成 ≥90%、动作事件 ≥90%、命令联合通过 ≥80%”计算，当前 {candidate["passed_classes"]}/11 类达标。独立来源自然动作严格跟踪通过 {fraction(cn["accurate_complete"],cn["requests"])}；新的桌面流程通过 {fraction(table_result["success"],table_result["planned"])}。这些结果限定了现阶段的覆盖范围，不能把少数 demo 当成通用动作能力。')
table([['本轮工作','事实与边界'],['共享策略','联合 PPO；部署端仅一个 MLP，动作类别不进入策略输入。'],['保留旧能力','训练中使用一个冻结稳定 actor 的回放 MSE；有保留/蒸馏正则，但没有为每个新类别另训教师。'],['类别扩展','联合语料由 777 条扩到 1,084 条，加入 307 条自然动作训练来源；扩展候选按统一开发规则比较。'],['生成端','本轮保持已有 FrankenMotion 及命令适配器冻结；不能称为整条 pipeline 只有一个权重。'],['交付与备份','旧 tracker、适配器和 demo 已备份；新权重、源代码、曲线、视频与原始记录独立保存，未 push。']],[94,401])
p('实验窗口为北京时间 02:19–10:19，共 8 小时。这个报告记录已完成的实验及失败；仍需独立全流程的多训练随机种子、更多运动来源和真机验证，才能形成完整论文证据。','small')
page();heading('1  实际 pipeline 与权重继承')
figure(D/'figures/architecture/pipeline_zh.png')
p(f'选中 actor 的输入为 {dim} 维，隐藏层 512/256/128，ELU，输出 29 维归一化关节动作。当前观测 160 维；每组未来参考为 67 维，偏移为 {frozen["preview_offsets"]} 个 50 Hz 控制步。TorchScript 包含固定观测归一化，导出与原 checkpoint 已做数值一致性校验。')
p('当前观测包含参考关节位置/速度、torso 锚点的局部位置和朝向误差、IMU 速度、关节状态与上一动作。归一化动作经固定 scale/offset 转为关节目标，再由原生 PD 执行；原生评估控制频率 50 Hz、物理步长 5 ms。tracker 不直接接收桌面几何或接触事件；场景位置用于上游命令和参考放置，接触由物理仿真实际产生。')
p('控制框架和权重继承 BeyondMimic 系列，历史稳定策略含 SONIC 等教师来源。本轮在此基础上联合训练；不是从零训练，也不是与已有方法无关。部署不包含训练 critic 或教师。')
p('“一套权重”仅指部署 tracker。11 类基准的重定向沿用冻结配置：举手和前倾保留由人体源姿态导出的腕相对位置 / 躯干方向约束；桌面 reach 沿用腕部加权 GMR。它们不是本轮新加的命令目标编辑，但包含任务相关的参考处理，不能把整个 pipeline 宣称为完全无任务分支。','small')
p('当前 demo 先离线生成和重定向整段参考，再进行物理控制。50 Hz 是仿真控制频率，1× 是视频回放速度；本轮未验证生成—重定向—执行的在线端到端时延。','small')
p('GMR 与参考管理包括已声明的整段行走重定时、刚性坐标放置、片段过渡和最终站姿。相对命令边界按机器人实际根姿态调整未来参考坐标；未反馈给扩散模型重生成，不属于完整生成器闭环重规划。当前 G1 为 29 DoF 机身加固定手部碰撞体，未验证 Dex3 灵巧手抓取。')
page();heading('2  数据隔离与评价标准')
table([['数据','规模','使用方式'],['基础联合训练','777 条 / 349,073 帧','11 类、既有组合/序列与 42 条桌面训练流程。'],['扩展联合训练','1,084 条 / 452,025 帧','加入 307 个自然动作训练来源；默认采样 15% 桌面、25% 新动作、60% 旧组。'],['历史开发','110 条参数动作 + 6 条桌面流程','用于诊断、选择全局 checkpoint；不是最终盲测。'],['新噪声测试','880 条 =11×4×4×5','已有文本模板 / 4 新噪声 / 5 数值命令。冻结后生成，非未见语言。'],['独立来源测试',str(cn['requests'])+' 条自然动作','每类预先取前 5 个候选、按源文件跨类别去重；无训练/验证来源重叠。'],['新桌面测试',str(table_result['planned'])+' 条','6 个随机布局，0.3/0.5 m reach 成对；固定桌高 0.8 m。']],[95,143,257])
p('自然动作的“严格跟踪通过”要求完整执行、全局平均身体位置误差 ≤0.10 m、逐帧平均误差 P95 ≤0.25 m。它不是动作语义成功，也不是生成模型的泛化结果：这些测试输入是官方标注 mocap 参考。来源隔离只针对本次 tracker 实验，不能保证基础模型预训练从未见过相关数据。')
p('完成判定沿用直立跟踪协议：骨盆高度 <0.35 m、根部俯仰/横滚合量 >60° 或状态非有限时终止。因此终止不全等于摔倒；深蹲和低姿态参考可能超出当前评价范围。冻结测试不临时放宽这些阈值。','small')
p('所有失败保留。物理完成不等于动作事件发生；事件发生也不等于数值准确。语义误差按命令范围归一化，截断到 1，摔倒或事件失败计 1。曲线只绘制可测量数量，缺失不补零；因此需要结合全请求通过率阅读。')
p('宏观差值置信区间按共同文本模板/噪声块重采样，保留该块所有 11 类与五个命令值，避免把相关的命令样本当成独立观测。它只反映测试来源不确定性，不是多次训练种子的方差。CPU、GPU 两个执行协议分开报告。')
page();heading('3  880 条冻结测试：同一原生物理模型')
table([['策略','物理完成','动作事件','命令联合通过','语义误差↓'],*[[label,fraction(c['aggregate']['complete'],N),fraction(c['aggregate']['event'],N),fraction(c['aggregate']['joint'],N),f'{c["aggregate"]["macro_semantic_E_all"]:.4f}'] for label,c in [('旧稳定版',stable),('旧广覆盖共享版',broad),('本轮共享版',candidate)]]],[99,100,100,105,91])
p(f'相对旧广覆盖版，联合通过率差值为 {delta[2]*100:+.2f} 个百分点，共同来源块 bootstrap 95% 区间 [{ci[2][0]*100:+.2f}, {ci[2][1]*100:+.2f}]；语义误差差值 {delta[3]:+.4f}，区间 [{ci[3][0]:+.4f}, {ci[3][1]:+.4f}]。误差越低越好。')
names=dict(raise_hand='举手高度',reach='前伸距离',strike='出拳速度',wave='挥手幅度',turn='右转角度',sidestep='右侧步距离',back_walk='后退速度',kick='右踢幅度',jump='跳跃高度',lean='保持前倾角度',walk='前行速度')
table([['类别','完成','事件','联合通过','达标'],*[[names[t],f'{s["actual"]["measurable"]}/80',f'{s["actual"]["event_pass"]}/80',f'{s["actual"]["joint_pass"]}/80','是' if candidate['class_pass'][t] else '否'] for t,s in candidate['per_task'].items()]],[160,80,80,100,75])
p('SONIC 单独采用 release mode 0 的原部署模型和 PD，不与上表混为同一执行器条件。它是原生系统级对照，不能将全部差距归因于网络结构，也不代表复现官方论文评分。','small')
if data['sonic']['audit']:
 sa=data['sonic']['audit']['aggregate'];p(f'SONIC 原生系统对照：完成 {sa["complete"]}/{N}，事件 {sa["event"]}/{N}，联合通过 {sa["joint"]}/{N}，语义误差 {sa["macro_semantic_E_all"]:.4f}。','small')
page();heading('4  命令在生成、重定向、执行三层的保留')
table([['类别','人体参考联合通过','G1 参考联合通过','实际执行联合通过'],*[[names[t],f'{v["human"]["joint_pass"]}/80',f'{v["g1"]["joint_pass"]}/80',f'{v["actual"]["joint_pass"]}/80'] for t,v in candidate['per_task'].items()]],[155,115,115,110])
p('每层均同时检查动作事件与数值精度。人体参考已经偏离命令时，不能将所有误差归咎于 tracker；G1 参考合格、执行不合格时，则需要进一步区分参考动态可行性与控制能力。这是配对描述，不能单凭前后差值作因果归因。执行偶尔抵消参考偏差，也不等于改善了生成器。')
p('数量定义也有区别：举手 / 前伸是相对骨盆的腕位置 P95，挥手为横向 (P95−P05)/2；前行是路径长度除以时长，后退是净位移速度；跳跃为骨盆高度增量，踢腿为相对初始的踝前伸幅度。动作事件是冻结的几何代理，跳跃事件本身并非接触力检测。完整定义见 metric_definitions.json。','small')
p('11 类固定数值容差：举手 / 前伸 0.02 m；出拳 0.10 m/s；挥手 0.015 m；转向 0.08 rad；侧步 0.06 m；后退 0.05 m/s；踢腿 0.05 m；跳跃 0.04 m；前倾 0.06 rad；前行 0.05 m/s。角度保持 rad，距离和速度为人体等效单位。')
p('桌面任务采用原有流程标准，与 11 类单动作容差不同：reach 误差 ≤0.08 m、保持阶段根位置误差 <0.15 m、连续桌面顶部接触 ≥0.5 s、收手下降 >0.10 m、转向误差 ≤12°、离开位移误差 ≤0.25 m、最终参考端点误差 ≤0.30 m，并须完整执行。接触每 20 ms 检查，含法向 / 位置 / 接触力 >0.2 N 条件；不代表全过程高频接触验证。')
for title,folder in [('上肢参数响应','final_upperbody'),('位移与转向响应','final_locomotion'),('动态动作与姿态响应','final_dynamic')]:
 page();heading('5  '+title);figure(D/'figures'/folder/'command_response.png');p('灰：人体生成参考；蓝：G1 参考；紫：旧共享 tracker；橙：本轮共享 tracker。若有绿色，为 SONIC 原生部署对照。阴影为来源 P10–P90，不是置信区间；数字表示可测量 / 计划样本数。角度保持 rad，距离与速度换算成人体等效单位。','small')
 p('接近命令对角线表示数值准确；斜率偏小表示命令变化在下游被压缩。即使某条曲线有良好斜率，也必须满足对应动作事件；曲线不能替代上一页的联合通过率。')
page();heading('6  自然动作扩展与稳健性')
table([['自然动作','请求','旧共享严格通过','本轮完整执行','本轮严格通过'],*[[t,v['requests'],broad['natural']['results'][t]['accurate_complete'],v['complete'],v['accurate_complete']] for t,v in candidate['natural']['results'].items()]],[143,58,104,94,96])
if data.get('natural_termination'):
 incompatible=sum(r['reference_ever_below_0p35'] or r['reference_ever_tilt_over_60'] for r in data['natural_termination']);p(f'其中 {incompatible}/{cn["requests"]} 条参考自身曾越过当前直立协议的高度 / 倾角边界；它们保留在分母中，不能把这些终止直接解释为机器人摔倒。','small')
p(f'自然动作严格跟踪：本轮 {cn["accurate_complete"]}/{cn["requests"]}，旧共享版 {natural_totals(broad)["accurate_complete"]}/{cn["requests"]}。本轮未提高这项独立来源指标，不能宣称自然动作泛化已改善。')
p('扩展训练与最后部署选择是两件事：'+('本次选中的是扩展语料联合训练的候选。' if selected.startswith('v5_') else '扩展候选已进入统一比较，但最终选择仍是保留旧能力较好的候选；没有为新动作另换权重。'))
npair=data['diagnostics'].get('natural_paired')
if npair:
 am=npair['candidate_means'];bm=npair['broad_means'];p(f'事后误差分解：两者都完成的 {npair["jointly_completed"]} 条中，本轮 / 旧版平均全局身体误差为 {am["global_mpjpe_m"]:.3f} / {bm["global_mpjpe_m"]:.3f} m，根相对误差为 {am["root_aligned_mpjpe_m"]:.3f} / {bm["root_aligned_mpjpe_m"]:.3f} m。共同完成子集的平均误差较低，不等于逐条严格通过数提高；原标准与失败分母保持不变。','small')
page();heading('6  物理扰动与跨后端复核')
robust_rows=[['原生 CPU 110 条子集','旧共享联合通过','本轮联合通过','本轮物理完成']]
for profile,label in [('nominal','名义条件'),('friction_0p6','摩擦系数 ×0.6'),('mass_1p1','机身质量/惯量 ×1.1'),('delay_20ms','执行目标延迟 20 ms'),('lateral_push_40N','世界 Y 方向 40 N，持续 0.1 s')]:
 a=data['robustness']['candidate'].get(profile);b=data['robustness']['broad'].get(profile)
 if a and b:robust_rows.append([label,str(b['aggregate']['joint'])+'/110',str(a['aggregate']['joint'])+'/110',str(a['aggregate']['complete'])+'/110'])
table(robust_rows,[185,105,105,100]);p('扰动每次只改变一个因素，未据此重新训练。推力固定作用于仿真第 2.0–2.1 秒；延迟为执行器目标延迟，不是所有感知通道延迟。这些测试不能替代真实硬件验证。','small')
if (D/'figures/robustness/robustness.png').exists():figure(D/'figures/robustness/robustness.png')
if data['gpu']['candidate'] and data['gpu']['broad']:
 ga=data['gpu']['candidate']['aggregate'];gb=data['gpu']['broad']['aggregate'];p(f'独立 GPU 协议的预先指定 110 条子集：本轮完成 {ga["complete"]}、联合通过 {ga["joint"]}、语义误差 {ga["macro_semantic_E_all"]:.4f}；旧共享版分别为 {gb["complete"]}、{gb["joint"]}、{gb["macro_semantic_E_all"]:.4f}。与原生 CPU 数值不混合。','small')
page();heading('7  为什么命令会在下游消失')
figure(D/'figures/root_observability/root_observability.png')
p('当前 SONIC mode 0 所需输入包含未来关节位置、速度和朝向，没有根水平位移。两个开发来源中，保持关节/朝向/时间不变，仅将根 XY 位移缩放为 0.5/1/1.5，SONIC 实际轨迹逐帧相同；带锚点位移输入的 v4 策略会改变位移。结论只针对这个接口，不是 SONIC 的所有控制模式。')
p('这是人为改根轨迹的反事实诊断，不是生成命令 demo。放大根位移而不改变步态会增加接触点切向速度代理；因此“对位移敏感”不等于“物理质量好”。正式 demo 的参数来自生成条件，不能用这种事后缩放冒充。')
page();heading('8  参考物理一致性：正面证据与失败尝试')
figure(D/'figures/aerial_dynamics/aerial_dynamics.png')
p('在离地且无外部接触的阶段，整个机器人的重心应按重力加速。部分参考却在预蹲/落地附近悬空，或存在不符合重力的加速度。原参考腾空残差约 2.8–10.7 m/s²；真实执行的可测腾空段约 0.01–0.04 m/s²。这里只是有限差分诊断，不是完整可行性证明。')
p('统一支撑贴地与弹道重心修正将参考残差降到约 0.09–0.12 m/s²，但固定 v4 的跳跃完成从 10/10 降到 9/10，命令联合通过仍为 0。这个方法未纳入主流程。只改腾空部分没有解决地面发力、角动量、摩擦/力矩限幅和落地，同时也改变了策略见过的参考分布。')
p('自然动作也存在类似问题：57 条开发参考中 30 条包含足底离地超过 2 cm、无外部接触且重心重力残差 >2 m/s² 的持续片段。仅对异常支撑做统一、有限幅度的根高度修正，保持关节、根 XY、朝向与时长不变，固定 v4 的完成数由 14/57 到 16/57；相对原始参考的严格跟踪仍为 6/57。未修改的 27 条执行逐帧完全一致。这个诊断未纳入最终流程，也不能证明所有失败由参考造成。')
page();heading('9  开发选择、消融与失败记录')
dev=[['开发候选','11 类联合通过','桌面通过','躯干抖动 / 稳定版']]
dev_labels={'baseline_stable':'旧稳定版','baseline_broad':'旧广覆盖共享版','v3_3000':'v3：联合基础版','v4_3000':'v4：长预览与物理平滑','v4_final':'v4：最后 checkpoint','v6_final':'v6：保守微调','v6_2000':'v6：保守版 2000','v7_2000':'v7：保守版＋平滑项','v8_2000':'v8：保守版＋合法重置','v9_3000':'v9：匹配短预览对照'}
seen_dev=set()
for name in ['baseline_stable','baseline_broad','v3_3000','v4_3000','v4_final','v6_2000','v6_final','v7_2000','v8_2000','v9_3000',selected]:
 if name in seen_dev:continue
 seen_dev.add(name)
 row=next((r for r in data['development_selection']['results'] if r['name']==name),None)
 if row and row.get('eleven'):dev.append([dev_labels.get(name,name+'：扩展联合版'),str(row['eleven']['joint'])+'/110',str(row['table_success'])+'/'+str(row['table_planned']),f'{row["walk_torso_jitter_ratio"]:.3f}'])
table(dev,[145,112,90,148])
p('开发资格门槛为桌面 6/6、躯干 5 Hz 高通角速度 RMS ≤旧稳定版 1.5 倍、11 类联合通过不少于旧广覆盖版 49/110、每类完成率 ≥80%。合格者按 11 类命令通过率与 12 类自然动作严格通过率的等类别均值排序；这个混合分数只用于选模型，最终结果分开报告。规则在扩展训练结果产生前确定，但不是整个探索开始前的预注册。')
p('v3 的命令分数较好，却约有 3 倍躯干高频抖动，未作为平稳 demo 权重。长预览、物理平滑项和合法重置相位分别安排了匹配对照；全部仍为单训练种子，不能仅凭一次结果断言普遍有效。扩展训练是从已有候选继续优化，不是等初始化、等预算的数据量消融。')
diag=data['diagnostics'].get('seen_unseen')
if diag:
 r=next((r for r in diag['results'] if r['name']=='training_reference_v5_final'),None)
 if r:p(f'扩展训练最后 checkpoint 的 60 条已见训练来源诊断：完整执行 {r["complete"]}/60、严格跟踪 {r["accurate_complete"]}/60。初始化直接使用参考姿态 / 速度；仍有较多失败，说明不能把问题全部归结为未见来源的泛化。这是已见训练来源的辅助诊断，不计入独立测试分数。','small')
p('匹配迭代对照中，v6_2000 / v7_2000 的命令联合通过为 49/110 / 38/110，桌面为 4/6 / 5/6；单加物理平滑项并非全面改善。v8 合法重置为 48/110、桌面 3/6。长预览 v4_3000 对短预览 v9_3000 为 53/110 对 51/110，桌面均 6/6，抖动比分别 0.989 与 1.130。这些是一次训练的开发对照，仍不足以给出稳定的因果结论。','small')
p('其他负结果：残差低通没有改善抖动；自碰撞 IK 减少部分穿透但未改善整体指标；reach 高度输入增加 6 cm 时掌心高度只变化约 2.6 mm，不能宣称可变桌高；“边走边挥手”生成未稳定包含两个动作。拍手候选出现手与对侧手腕接触，未作为掌心对掌心成功展示。')
page();heading('9  扩展联合训练：配置与开发曲线')
figure(D/'figures/expanded_training/expanded_training.png')
p('曲线只使用开发集，最后点是主候选；自然动作列为严格跟踪，命令列为事件与数值联合通过。不同 checkpoint 均经过相同桌面与抖动资格检查。独立自然测试没有复现开发分数的提升，因此不把这条曲线当泛化证明。','small')
table([['配置','本次实际设置'],['采样与预算','260 环境 ×24 步 ×6,000 次更新，约 3,744 万环境控制步；39 个桌面槽位、65 个扩展槽位、156 个旧动作槽位。'],['优化器','Adam；固定学习率 2e-5；PPO clip 0.2；5 epochs /4 minibatches；γ=0.99，λ=0.95，entropy=0.001。'],['参考观察','当前状态＋未来 0.10/0.20/0.40/0.70/1.00 s；共享输入归一化，无任务类别输入。'],['重置与保留','35% 片段起点，其余均匀参考相位；保留回放 53,644 条训练观测，每次优化采样 256 条，MSE 权重 0.5。'],['共同奖励','全局根位置 / 朝向、身体与关节跟踪、骨盆相对末端位置、速度和双足高度。全部动作使用相同公式与系数。']],[90,405])
p('平滑项作用于物理关节目标变化与参考关节速度之间的残差，同时保留少量标准动作变化惩罚。它是训练奖励，不是部署输出上的逐类滤波。当前最终版本从 v4_3000 继续训练，不能把上述预算当作从零训练的总成本。','small')
replication=data.get('replication',{})
if replication.get('results'):
 page();heading('9  扩展阶段的第二个训练随机种子')
 rr=replication['results'];rn=natural_totals(rr);rt=rr['table'];p('在最终测试结果产生前安排独立 seed 6107：使用相同 v4 起点、相同扩展语料、奖励、保留正则和并行环境数，训练到主候选选定的同一个迭代数。不为第二个 seed 重新挑 checkpoint，也不替换主候选。')
 table([['固定测试','主候选','第二个扩展 seed'],['11 类物理完成',fraction(agg['complete'],N),fraction(rr['aggregate']['complete'],N)],['11 类命令联合通过',fraction(agg['joint'],N),fraction(rr['aggregate']['joint'],N)],['语义误差（越低越好）',f'{agg["macro_semantic_E_all"]:.4f}',f'{rr["aggregate"]["macro_semantic_E_all"]:.4f}'],['自然动作严格跟踪',fraction(cn['accurate_complete'],cn['requests']),fraction(rn['accurate_complete'],rn['requests'])],['桌面完整流程',fraction(table_result['success'],table_result['planned']),fraction(rt['success'],rt['planned'])]],[200,147,148])
 figure(D/'figures/seed_replication/seed_replication.png')
 p('总通过数接近仍会掩盖类别差异：主种子 / 第二种子的出拳联合通过为 41/80 / 0/80，侧步为 41/80 / 64/80，跳跃完整执行为 80/80 / 55/80。第二种子的出拳事件仍为 80/80，但速度整体偏低；0/80 指数值容差不达标，并非没有出拳。总体语义误差也高于旧共享版，因此误差下降与动态稳定性尚未跨种子复现。','small')
 p('两个策略共享扩展之前的初始化权重。因此这里只检验扩展阶段训练随机性，不是从数据、预训练到控制训练全部独立的多种子结果。两个 seed 也不足以可靠估计训练方差；不得把测试来源 bootstrap 区间当成训练随机性的区间。')
elif replication.get('status'):p('附加扩展阶段第二个 seed 尚未完成全部固定评估，当前状态为 '+str(replication['status']['stage'])+'；不插入替代分数。','small')
study=data.get('projection_study')
if study and study.get('projection_audit'):
 page();heading('9  探索性对照：共享策略生成物理训练参考')
 pa=study['projection_audit'];p(f'冻结正式测试后追加的假设探索：同一个主策略执行全部 307 条新增训练来源；完整执行 {pa["physical_complete"]} 条，其中 {pa["accepted"]} 条同时满足平均全局身体偏差 ≤0.20 m、逐帧平均偏差 P95 ≤0.40 m。只用这些实际轨迹替换对应训练参考，其余保留原参考。没有按动作另训教师。')
 p('两支训练使用相同主策略起点、seed 6108、2,000 次更新、优化器、采样比例与奖励。对照支继续使用原始 1,084 条参考；实验支仅替换上述通过筛选的参考，片段数量与时长不变。只评估开发集，不重选正式权重，不再次使用最终测试挑结果。')
 if study.get('results'):
  rr=study['results']['results'];rows=[['开发评估','原始起点','原参考续训','物理参考续训']]
  rows.append(['11类联合通过 /110',*[str(rr[tag]['eleven']['joint']) for tag in ['initial','raw','projected']]])
  rows.append(['自然动作严格通过 /57',*[str(natural_totals(rr[tag])['accurate_complete']) for tag in ['initial','raw','projected']]])
  rows.append(['桌面通过 /6',*[str(rr[tag]['table']['success']) for tag in ['initial','raw','projected']]])
  table(rows,[200,98,98,99]);p('本次对照没有显示物理参考替换的收益：自然动作严格通过从原参考续训的 12/57 降到 9/57，11类联合通过从 63/110 降到 57/110。两支桌面均为 6/6。这是单种子、小比例参考替换的负结果，不足以否定更完善的物理参考训练方法。');p('这是主测试结果已知之后的一次开发探索，不是新的确认性证据。原始起点的11类数值来自原生 CPU 协议，不能与前面开发曲线的 GPU 分数混用。')
 else:p('本对照尚未完整完成：'+str((study.get('status') or {}).get('stage','unknown'))+'。不将部分训练或部分评价替代完整对照结果。')
 p('物理轨迹来自当前仿真模型，仍可能含命令偏差或不自然动作；执行成功不构成语义准确的保证。筛选会偏向教师已经能执行的动作，只能作为后续统一物理参考训练的初步测试，不能等价为 OmniTrack 的完整复现。')
page();heading('10  Demo、命令含义与复现入口')
p(f'所有正式展示使用冻结 checkpoint {selected}。随机桌面全部 {table_result["planned"]} 条结果保留，成功 {table_result["success"]} 条；展示片段在完整评估后按物理结果与视觉检查选择，不代表所有布局成功。具体视频、场景和参数见交付目录的视频索引。')
table([['流程 / 参数','具体含义'],['走到桌边','方向和距离进入生成条件；对同一噪声做最多四次输入修正，再经 GMR 和已声明的行走重定时。不是事后拉伸根轨迹。'],['reach 0.3 / 0.5 m','人体等效的腕部前伸命令，不是机器人世界坐标中的绝对位置。两条配对流程使用同布局与同噪声。'],['收手、转向、离开','reach 生成片段含放手/收回阶段，后接生成的 90° 或 135° 转向与 1.2 m 前行；片段间有显式过渡。'],['转向外推','90°/135° 超过原名义 0.45–1.5 rad 范围，使用同噪声命令输入修正；不能写成训练范围内一次采样即精确。'],['其他展示','行走—指向—鞠躬—转身离开，以及走停转向与挥手。指向与鞠躬为文本条件动作，没有数值幅度命令。']],[100,395])
p('服务动作展示使用已有生成片段组合，鞠躬取官方文本生成结果的 20 Hz 第 80–143 帧，省去初始 T-pose 与多余转身。保留片段内部原姿态，使用已声明的刚性放置与过渡；这是展示用时间裁剪，不是一次生成整段服务任务。另试的四条简短鞠躬 / 弯腰提示均未产生明显前屈，虽物理完成仍作为语义失败保留。','small')
if data.get('table_quality'):
 qrows=[['新桌面流程','完成 / 请求','躯干高频 RMS (rad/s)','足部切向速度代理']]
 for tag,label in [('stable','旧稳定版'),('broad','旧共享版'),('candidate','本轮共享版')]:
  q=data['table_quality']['controllers'][tag];m=q['completed_only_means'];j=m['walk_torso_angular_velocity_highpass5hz_rms'];v=m['walk_foot_contact_tangent_mean_m_s'];qrows.append([label,f'{q["complete"]}/{q["planned"]}',f'{j:.4f}' if j is not None else '不可测',f'{v:.4f} m/s' if v is not None else '不可测'])
 paired=data['table_quality'].get('paired_common_completed',{}).get('stable')
 if paired:
  key='walk_torso_angular_velocity_highpass5hz_rms';p(f'在两者都完成的相同 {paired["requests"]} 个场景上，本轮 / 旧稳定版的步行躯干高频 RMS 为 {paired["candidate"][key]:.4f} / {paired["baseline"][key]:.4f} rad/s。这个配对子集仍以完成为条件。','small')
 table(qrows,[111,94,138,152]);p('质量均值只针对完整执行的流程；各策略完成的集合可能不同，不能忽略上表分母。足部指标是几何接触点切向速度代理，不是基于接触力的滑移或真机测量。','small')
p('另有一条走近—指向—后退—侧移—转身离开的探索组合完整执行。后退命令 0.625 m/s，实测约 0.597；侧移命令 0.8 m，实测约 0.749 m，但朝向变化 0.5541 rad 略超固定 0.55 rad 阈值，按原规则记为未通过，单独存为诊断视频。不能把完整执行当作全部命令达标。','small')
p('视觉核验使用实际物理轨迹的关键帧与密集步态帧，并结合抖动、接触、命令误差检查；不是只看参考回放。视频为 1×速度，图中参考回放明确标注，不替换机器人真实状态。')
p('远程实验：/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006\n源代码：/home/pku/frankenmotion/work/universal_tracker_20261006\n原权重备份：backup/stable_frozen 与 backup/prior_unified_frozen\n新权重：frozen_unified/policy.pt 和 actor.pt\n训练参数、原始轨迹、失败日志、来源与 SHA256 清单均保留。','small')
p('冻结 checkpoint SHA256：'+frozen['checkpoint_sha256'],'small')
if data.get('long_horizon_ablation'):
 page();heading('10  长时组合：相对命令与固定世界参考')
 p('同六段生成动作重复四轮，形成24段命令、134.2 s参考。采用相同冻结权重、相同参考与初始化，只改变是否在阶段边界按实际位置和朝向重新放置后续参考。两种设置均连续完成，不重置物理状态。这是正式测试后的单组探索对照。')
 figure(D/'figures/long_horizon/long_horizon.png')
 lh=data['long_horizon_ablation']['results'];ar=lh['relative_anchors'];fw=lh['fixed_world_reference']
 table([['单次长流程指标','相对参考锚定','固定世界参考'],['完整命令段',str(ar['completed_stages'])+'/24',str(fw['completed_stages'])+'/24'],['相对初始世界路线的根XY RMS',f"{ar['initial_world_plan_root_xy_rmse_m']:.3f} m",f"{fw['initial_world_plan_root_xy_rmse_m']:.3f} m"],['相对初始世界路线的终点误差',f"{ar['initial_world_plan_final_root_xy_error_m']:.3f} m",f"{fw['initial_world_plan_final_root_xy_error_m']:.3f} m"],['单段步行距离最大误差',f"{ar['walk_distance_max_error_m']:.3f} m",f"{fw['walk_distance_max_error_m']:.3f} m"],['单段步行方向最大误差',f"{ar['walk_direction_max_error_deg']:.2f}°",f"{fw['walk_direction_max_error_deg']:.2f}°"],['单段转向最大误差',f"{ar['turn_max_error_deg']:.2f}°",f"{fw['turn_max_error_deg']:.2f}°"]],[245,125,125])
 p('相对锚定的用途是从当前位置开始下一条命令，因此它会主动改变后续世界路线。上表对初始路线的偏差不等于对更新后参考的跟踪误差。固定世界参考在此例中也能稳定执行，说明连接全局导航时可以保留世界目标约束，不必一律重锚定。不能据单例推断所有长时任务都适用。')
 if data.get('long_horizon_delay'):
  dl=data['long_horizon_delay'];p(f'延迟敏感性补充：固定世界参考模式加入20 ms关节目标执行延迟后，在 {dl["termination_time"]:.2f} s 触发终止，完整执行 {dl["completed_stages"]}/24 段。因此理想条件下的长时成功不能作为真机就绪的依据；本轮未针对该结果调整权重。这是单次探索，不是完整硬件延迟模型。','small')
 p('六个生成片段被重复使用，24段不是独立测试样本，也没有增加新的动作类别。首次两个运行器尝试在读取输入时因缺少原生motion_path报错，未开始物理执行；补齐标准FK转换后按固定参考各执行一次，未更改权重或挑选轨迹。','small')
page();heading('11  论文定位与下一步')
p('本轮有实证支撑的问题是：参数命令在生成、重定向、物理执行三层分别保留了多少；为什么增加全局响应会与平滑性或接触质量冲突；参考自身的物理缺陷如何限制共享控制器。一个策略支持多个动作本身已有工作，不能单独作为新颖性主张。')
p('下一步应优先补齐全局训练的数据与参考质量：统一处理支撑/接触与动态可行性，保留命令量作为约束；再用同一训练机制覆盖更大动作库。物理一致参考也可能把命令幅度压小，因此不能只把现有策略执行结果当成教师而不检查命令。随后补多种训练随机种子、未见文本/动作来源、长时组合和匹配的真机接口实验。')
p('相关工作与本实现的关系：OmniTrack 先用特权通用策略产生物理一致参考，再训练通用控制器；CLOT 针对全局漂移与激进纠偏引入专门训练设计；Robust and Generalized Humanoid Motion Tracking 使用时序本体感知与条件命令聚合。本轮没有复现这些方法，简单弹道修正也不等价于它们。')
for label,url in [('FrankenMotion','https://arxiv.org/abs/2601.10909'),('BeyondMimic','https://beyondmimic.github.io/'),('SONIC','https://nvlabs.github.io/GEAR-SONIC/'),('BABEL','https://babel.is.tue.mpg.de/data.html'),('OmniTrack','https://arxiv.org/abs/2602.23832'),('CLOT','https://arxiv.org/abs/2602.15060'),('Robust and Generalized Humanoid Motion Tracking','https://arxiv.org/abs/2601.23080')]:p(label+'：'+url,'small')
p('资源与结论边界：单台 RTX 4090 与同机常驻进程共享显存，本轮训练使用 130 或 260 个并行环境。有限时间下的负结果不能证明模型架构本身无法扩展；所有训练预算与采样计数已记录。','small')
p('现有局限：主要对照为单训练种子，附加 seed 只重跑扩展阶段；固定桌高、无视觉感知闭环、无灵巧手抓取、无障碍地形和真机验证。实验协议和证据按可复现研究整理，但不等于已经完成可发表论文。')
if data['missing_optional_files']:p('尚缺的附加结果文件：'+', '.join(data['missing_optional_files']),'small')
def footer(c,doc):
 c.setFont('Chinese',8);c.setFillColor(colors.HexColor('#607487'));c.drawString(50,27,'FrankenMotion / G1 | 冻结测试与研究记录');c.drawRightString(545,27,str(doc.page))
output=OUT/'统一Tracker扩展实验报告_中文.pdf';doc=SimpleDocTemplate(str(output),pagesize=(595.28,841.89),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=47,title='FrankenMotion G1 统一Tracker联合训练与扩展实验',author='实验记录');doc.build(story,onFirstPage=footer,onLaterPages=footer);print(output)
