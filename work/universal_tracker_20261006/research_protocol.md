# 2026-10-06 统一 tracker 扩展

授权窗口：北京时间 02:19 至 10:19，8 小时持续实验。旧成果备份后不覆盖，不 push。

研究问题：单一参考条件策略能否同时保留多类动作的数值响应、平滑步态与连续交互能力？当前工作基于 BeyondMimic 框架/预训练权重及已有 SONIC 教师蒸馏策略，不将基础控制能力或联合 PPO 本身声明为原创。

## 对照与训练

- 基线：当前桌面稳定 actor；之前 11 类共享 actor。历史按任务选择策略的版本仅作为明确标注的参考上限，不是统一部署。
- 第一组：当前稳定 actor 初始化，735 条训练参考的统一 PPO，不加入保留约束。
- 后续组：统一混合语料，包含桌面训练序列；所有动作共享参数、观测、奖励公式与动作接口。尝试全局骨盆/末端跟踪和参考速度约束；根据开发集决定是否添加旧能力回放。禁止逐类训练后挑选策略拼装。
- 不停止服务器上其他模型服务；在当前可用显存下调度。

## 数据隔离与评估

既有 110 条开发集用于诊断和统一 checkpoint 选择。历史 880 条最终集仅作已公开的历史结果说明，不宣称为本轮独立盲测。新增最终集使用新噪声，单一 checkpoint 固定并记录哈希后才评估，不能看完后挑 checkpoint 重测同一最终集。

报告人体参考、G1 参考、物理执行三个阶段。保留既有任务事件和命令容差，不为了达标修改规则。报告每类完成率、事件通过率、联合通过率、命令误差/斜率/单调性，另测连续组合、平滑性、脚部滑动和接触。

“11 类都很好”的工作门槛预先设为：每类物理完成率至少 90%，事件通过率至少 90%，联合命令通过率至少 80%；另检查响应曲线及退化，不能以宏平均遮盖弱项。未达到就明确报告，更多类别只能算探索性测试，不能包装成已实现通用性。

## 候选 demo 与资料

- 多段转向与方向/距离导航：展示命令组合和累计误差。
- 行走与挥手、指向等全身组合：同时检验上、下肢响应和连续切换。
- 不同高度/距离的双侧伸手及收回、蹲起/伸展：优先选可由现有生成器自然产生的动作；未训练的数值条件不冒称参数化控制。
- 动态动作（跳跃、快速出拳、踢腿）先做量化，失败如实报告，不只展示成功视频。
- 抓取搬运需要物体状态、受力与手指控制，当前不以摆姿冒充真实物体操作。

原始资料：
- https://beyondmimic.github.io/ ：跟踪与引导扩散控制，导航/避障示例。
- https://nvlabs.github.io/GEAR-SONIC/ ：统一全身控制与运动规划接口。
- https://babel.is.tue.mpg.de/data.html ：序列及逐帧动作标签。
- 远程官方 Frankenstein annotations：按源动作文件隔离 train/val/test，不能将相同原始动作的重叠裁剪视作独立样本。

每个 demo 必须使用实际物理轨迹、一个固定 checkpoint，记录生成输入、参考处理、相机与播放速度，并进行视觉检查。实验结果未经过多随机种子与真机验证时明确注明。

## 04:15 开发阶段补充（扩展语料训练结果尚未产生）

部署候选首先必须通过六条历史桌面序列，并且相同参考下的行走躯干角速度 5 Hz 高通 RMS 不超过稳定版的 1.5 倍。该门槛只是必要条件；仍需检查 11 类的数值与事件响应，不能每个动作单独挑 checkpoint。若没有候选同时达标，保留稳定版，明确交付实验候选而不声称已完成替换。

自然动作扩展集另报告“物理完整且跟踪准确”：完整执行、全局身体位置平均误差不超过 0.10 m、逐帧身体位置平均误差的 P95 不超过 0.25 m。该指标是参考跟踪指标，不等于语义任务成功。参考本身没有蹲下或拍手时，机器人跟踪参考也不能被算作完成蹲下或拍手。站立进入为主协议；参考状态初始化（RSI）单列作为诊断，不与主协议混合。

新探索发现与研究联系：CLOT 讨论直接施加强全局跟踪可能带来激进、不稳定修正，并使用其自身的随机化和运动先验处理；本实验目前没有实现或复现 CLOT，也不把相关方法据为原创。资料：https://arxiv.org/abs/2602.15060 。CLoSD 展示生成器与控制器闭环、连续任务的重要性，其机器人形态与此处 G1 不同：https://guytevet.github.io/CLoSD-page/ 。当前相对命令序列仅在段落边界对未来参考做刚性世界坐标校正，没有把执行状态反馈进扩散模型重生成，不能称为完整的 CLoSD 式生成闭环。

显存不足的扩展训练首轮在首次 PPO 更新前失败，已独立保留目录和日志；原配置错峰重试，不更换外部服务。控制输出的参考相对残差滤波已在开发集测试，alpha=1 与原仿真逐帧完全一致，alpha=0.8/0.6/0.4 均没有改善躯干抖动，当前不采用。所有这些选择均基于开发集。

## 桌面接触与高度输入诊断

冻结桌面参考的手部碰撞几何在保持阶段会穿入桌面约 4.7–5.5 cm，骨盆坐标回放验证无误。回查 `reach_adapter_v6_20261005.py` 后确认：旧生成训练显式包含 0.025 m 接触预压及随 reach 命令变化的高度校准。因此这不应直接描述为未知坐标 bug；该参考具有接触控制目标的性质，不能把它当作完全可实现的运动学姿态。

为检验减轻预压是否解决新 tracker 的收手退化，使用同种子、同水平 reach 命令，单独扫已有生成器的高度输入 0.84/0.86/0.88/0.90 m；后两者超出原训练范围，明确标为探索。保留原桌面评测不变，另存 16 条对照。GMR 使用桌面原有腕部权重 80、方向权重 5，不能将这一支误写为与普通动作完全相同的 GMR 配置。变化发生在生成器输入，不改生成后的手部姿态。

测量发现高度输入增加 6 cm，G1 参考掌心平均高度仅增加约 2.6 mm。现有权重的水平 reach 参数响应较好，不代表高度参数也已学好。这组输入调整没有修复 v4_1000 的收手失败，尚无证据把退化完全归因于预压。未来支持不同桌高需要单独验证或训练高度条件，当前 demo 不冒称已有该能力。

## 标注与自碰撞对照（开发阶段）

指向类初版筛选把 caption 中的 “starting point / at one point” 误当成指向动作。扩展训练开始前，改为要求官方 action/left_arm/right_arm 段出现指向证据；20 条训练样本中替换了 9 条，验证样本保持不变，源文件隔离重新核查。旧误匹配生成样本仍保留作审计，不能当成指向 demo。balance 类在本数据中包含受推后的平衡恢复，不应称作单腿平衡。

原参考部分行走与转向存在手臂和髋/大腿几何重叠，最深约 3-6 cm。统一自碰撞 IK 对照固定根、躯干与全部腿关节，仅优化臂关节；使用 Mink/DAqp 与原模型碰撞体。穿透状态的距离 Jacobian 经有限差分核查（最大误差约 2.1e-9），逐帧固定坐标保持位级一致。该方法是现有 IK 框架上的重定向对照，不是动作生成器或新 tracker，也不作为原创方法宣传。

同一 v3_3000 权重、原生 CPU、110 个开发请求：原参考 110/108/59（完成/事件/联合参数通过），修正参考 110/108/58。六条桌面都成功，但行走躯干高通 RMS 从 0.2805 增至 0.2928。故这组对照没有证明自碰撞是高频抖动主因，不将其并入主结果。早期数值固定约束有 1.2e-6 浮点漂移导致检查失败，失败输出保留；v2 每次 IK 积分后显式恢复固定坐标再检查。

v4_3000 联合训练在原参考六条桌面流程达到 6/6，行走躯干高通 RMS 0.0922（稳定版约 0.0933）。GPU 开发 110 条完成、107 条事件通过、53 条联合通过；CPU 对应 110/105/54，两种后端分开报告。该结果尚未达到所有 11 类的预定标准，且只是一组开发候选，不是冻结最终测试。自然扩展类别仍有较多失败，后续扩展训练采用一个共享权重与联合语料，不按类别分别蒸馏。

## 冻结 SONIC 部署对照与根平移可观测性

新增冻结 SONIC release mode0 CPU 部署基线：使用其现有官方模型/PD 桥接配置、同一 GMR 参考，原生模型与 BeyondMimic 系列执行模型不同，因此属于系统级原生部署对照，不能声称已控制全部动力学变量。开发 110 条为 110 完成、82 事件、34 联合参数通过。release/observation_config.yaml 的 g1 mode0 required_observations 只有关节位置、速度、未来朝向；官方 C++ GatherEncoderMode 确认模式为标量后填零，不是一位有效编码。

另外做了明确标注为诊断的“根平移反事实”：关节、朝向、时序完全固定，仅将两条 walk 的根 XY 轨迹缩放为 0.5/1/1.5。SONIC 两组均逐帧相同，共享 v4 则改变行走位移。这是当前模式的输入可观测性证据，不代表 SONIC 所有接口都不支持导航。这些修改过参考的片段不能包装为生成命令 demo；增加位移还可能增加脚滑，接触点切向速度另行报告。

扩展训练开始前改为从开发集通过桌面与抖动门槛的 v4_3000 继续联合微调 1084 条训练参考，全局学习率 2e-5。这个阶段是优化而非同初始化、同算力的数据单因素消融；早先显存失败的原配置与日志保留。另启动 v9 做短/长预览的对照：与 v4 相同初始化函数、奖励、260 环境、种子及回放样本/教师目标，输入只截去两个更远期预览。单个训练种子的这些消融不能代替多种子论文结论。

新最终集在冻结后生成 880 条新噪声，主比较采用相同原生 CPU 模型和阈值的候选/旧广覆盖/旧桌面稳定策略；历史 GPU 协议另外固定抽取 p0_s0 与 p2_s2 共110条交叉验证，不混合两类后端。该计划在新最终生成前已保存 final_test_plan.json。固定官方 SONIC 原生部署另列。所有最终结果不再用于调整权重。

## 组合动作的语义核查

早期 walk_wave 组合 8 条生成/跟踪样本不能仅凭仿真完成就作为“边走边挥手”成功：4 条波形样本的人体参考按冻结挥手事件指标只有0-1个完整周期，实际机器人也没有持续挥手。对应结果保留，并从候选成功 demo 中排除。新的生成条件对照显示，walk TaskControl 保留行走但不能可靠挥手；wave TaskControl 配合 RootControl 行走输入时能响应0.12/0.20m挥手幅度，却几乎不发生行走。关闭 TaskControl 的 RootControl-only 在某个噪声下出现小幅挥手，但速度响应偏低。内部 latent 指令残差增益0.1/0.25/0.5的探针也没有同时解决两者；0和1端点与原模式逐元素一致。没有将这些潜变量增益或失败组合纳入主pipeline，也没有把后处理姿态伪装成生成结果。

相比之下，连续“走路-转向-继续走路-停下挥手”由分别生成的动作通过明确的参考拼接组成，属于时序组合而非同时执行。两类能力分开命名。新自然动作验证的失败多发生在运动过程而非初始过渡，v4_3000的40余个失败中仅3个发生在1.2s前；未以放宽摔倒阈值来提高完成率。少数低姿态/大倾角参考超出当前直立评估阈值，应在扩展到地面动作时另行设计协议。

## Additional development diagnostics, 22:31 UTC

- v6 final (conservative, short preview) completes GPU110/110, event108, joint53, semanticE0.155263; table6/6 and torso jitter1.095×stable. Natural validation16/57 complete,6/57 accurate. It is globally eligible and currently ranks above v4_3000 under the pre-expansion equal-category rule. This is not a frozen selection yet. NativeCPU110/108/54,E0.153535.
- Natural v4 RSI diagnostic: initialize at reference pose/velocity rather than standing. Completion17/57 versus14/57, accurate6/57 unchanged. Entry transition is not the sole failure explanation. Keep main standing-entry protocol unchanged.
- Generated clap_10477_s1 has a near-hand motion but actual collision is left-hand against right-wrist, not hand-hand; geometry records16 contacts in initial broader check, strict hand-hand0. Do not call it successful palm-to-palm clapping. Wrist-distance-only diagnostics miss palm thickness and cannot establish semantics.
- Contact-free COM audit on10 original jump development references: 11-frame cubic derivative,50Hz,allfeet>2cm with no static external contact. Reference residual2.8–10.7m/s²; actual interior-flight residual~0.01–0.04m/s² where enough airborne frames exist. Large references hover during pre-crouch/landing; this is a real reference issue, not proof that target height itself is impossible.
- Generic support grounding/ballistic COM probe applies without task IDs to all110 references. It grounds support rootZ and retimes inferred flights (>8cm bothfoot clearance,>=4 frames) preserving nominal COM apex. No angular momentum,torque/friction,or takeoff/impact feasibility optimization. Reference aerial residual becomes~0.09–0.12m/s², but fixed v4 jump worsens: complete10→9,event8→7,joint0→0. Whole110complete110→109,event105→106,joint54→53. Not adopted. Demonstrates COM consistency alone is insufficient and changing references creates a policy distribution shift.
- Retimed jump lengths58/59 require an explicitly separate diagnostic metric wrapper; jump extrema/landing equations unchanged and parity checked against final-frame padding, no fixed-window or velocity metric changed. Main frozen benchmark references remain untouched. Failed preparation attempts and old metric shape-assert logs retained.

## Expansion phase and matched checks, before fresh test

- Final authorized-window deadline corrected to the exact goal creation timestamp +8h:2026-10-06 02:19:43UTC (10:19:43Beijing). The nominal frozen-policy deadline remains00:35UTC.
- v8_2000 valid-reset-phase ablation: GPU109complete106event48joint,E0.203127; table3/6,jitter1.402. No convincing improvement over matched conservative v6_2000 (109/102/49,table4/6).
- v9_3000 matched short-preview: GPU110/109/51,E0.144369; table6/6,jitter1.130; naturalval16complete7accurate/57. v4_3000 long counterpart110/107/53,E0.169312,table6/6,jitter0.989,natural6accurate. More preview is not unconditionally better. They share initial policy function,training seed,envcount,corpus,rewards and replay observations/targets (short truncates identical long-retention dataset).
- Expanded run successfully started after v9 finishes,fromv4_3000,1,084clips,260envs,globalLR2e-5,6000iterations. It is a staged joint optimization,not equal-init/equal-budget corpus-only ablation. First500 checkpoint table6/6,jitter1.157; naturalval18complete5accurate/57. This is preliminary,not selected.
- Before inspecting any expanded dev scores (CPU jobs may already have produced intermediate files), added complete GPU checkpoint grid500/1000/2000/3000/4000/5000/5999 to match CPU natural/table grid. Global selection rule unchanged. ExtraGPU queue06 PID778105.
- v6_final visual check: table0 keyframes and20 side-view gait frames from1–4s show plausible stepping without evident crossed legs or large torso sway. This is discrete visual inspection,not full-video viewing. Geometric foot-contact tangent-speed mean0.05207m/s,p95average0.20613 versusstable0.04999/0.23681; no force-weighted/hardware slip claim.
- Natural test metadata-only audit:54 unique sources after deterministic first5/category andcross-category deduplication; no training/validation source overlap. Motion content remains unloaded untilfreeze. Counts:run5,squat5,stretch4,dance5,punch5,throw4,clap5,point5,balance5,lunge2,bend5,sidestep4.
- Primary macro CI amended before freshgeneration to resample matched prompt/seed blocks acrossall11tasks,retaining5commands each. This preserves shared noise. Independent-within-task bootstrap remains sensitivity; no training-seed CI claim. Development only2blocks can yielddegenerateCI,not strong statistical evidence.
- Final supervisor PID744018 will freeze globally at00:35UTC (or recognize an earlier manualfreeze),prepare fresh880/natural54/table12,thenrunCPU/GPU/robustness/generateddemo phases. Does not mark goalcomplete or choose basedon final results. Own evaluator source compile checks passed.

## Late development checks, 23:48 UTC

- Expanded v5 checkpoints currently improve the global eligible score: 1000=56/110 command joint and6/57 natural accurate,2000=57 and8,3000=58 and8. All table6/6; v5_3000 torso jitter0.959721×stable. No frozen choice yet; remaining4000/5000/5999 remain in the declared grid.
- v5_1000 table scene0 dense side-view frames1–4s show plausible stepping with no obvious crossed legs or large torso sway. Discrete image inspection, not full-video viewing.
- Seen-source RSI diagnostic: v4_3000 training subset21/60 complete,7/60 accurate; v5_1000 23/60 and9/60. Matched-entry v5_1000 validation17/57 and7/57. These diagnostics never enter checkpoint selection. Failures on seen references mean unseen-source generalization cannot be the sole explanation. Validation RSI queue launched separately and waits forv5_3000/final.
- Natural support audit:30/57 historical validation references have full11-frame contact-free windows with foot bottom>2cm and mean gravity residual>2m/s²;3/57 have foot penetration>2cm for>10%frames. This is a diagnostic, not a feasibility certificate.
- Generic support projection probe (fixedv4,all57): adjust global bodyZ only on non-ballistic unsupported intervals or substantial foot penetration,cap12cm,Gaussian100ms,keepjoint/rootXY/orientation/time unchanged. Modified30references,27unmodified physicalrollouts exactly identical. Completion14→16,strictaccurateagainstoriginalstill6/57; two newlycomplete,none lost. Not adopted into frozen references or selection. Diagnostic assessor handlesfallsbefore1s entry as null post-entry error,notzero/success.
- Supervisor restarted while still waiting,PID788058 replaces744018. Deadline remains00:35UTC. Frozen generated-demo evaluation now starts immediatelyafterfreeze,independentlyof fresh generation; CPU/GPU/robust tests waitforfreshpreparation. Supervisorstilldoesnotcompletegoal.
- Queued final evidence assembly and candidate rendering waitforcompletedphases. Source/tensorverification checks allfinalcandidateexports exactlyidentical. Presentationcandidates are notautomaticallyapproved; actualvisualreview required. Frozen table quality includesjitter andgeometricfoot-contacttangentialspeed,completiondenominatorsretained.
- Final Chinese report now explicitly distinguishes11-task numeric tolerances fromtableflow thresholds (reach0.08m/contact0.5s etc),includeslayerwise generation/GMR/execution pass rates andnegative support probes. No unpublished final numbers inserted.

## Additional pre-freeze commitments, 00:04 UTC

- v5_4000 now leads eligible development ranking:57/110 command joint,9/57 natural accurate,table6/6. Remaining5000/final are still awaited; no freeze yet.
- Prepared two36.65/38.2s service sequences: generatedwalk→right-arm pointing→bow→turn→walkaway. Existing generated source clips, uniformGMR, same declared retiming/rigid placement/1s blends/final stance, no new teacher. These will be evaluated only withthefrozenpolicy and are presentation candidates,notunseen-language benchmarks. No actual push/ball/finger actuation is simulated. Frozen demo evaluator now includesservice_motion; rendererqueuewillneedmanualservicepreviewafterevaluation.
- Before any fresh-test result, declared a supplementary seed6107 repetition ofthe selected v5 continuation. Samev4warmstart,corpus,envcount,rewards,retention,learningrate,exact selected iteration+1updates; no within-replica checkpoint selection or deployment switch. It starts onlyafterglobalfreeze andfirstv5complete,stopsat02:05UTCifunfinished,andreportsCPU880/natural54/table12 separately. It doesnotestimatefull-pipeline training variance sincewarmstartis shared. Scriptreplicate_frozen_continuation.py,statusreplication_status.json,planreplication_plan.json;mainfinal_test_plan.json includescommitment.
- Main evidence collector mayfinishbeforetheseedreplica. Reruncollect_release_inventory.py andcollect_report_data.py afterreplicacompletionbeforefinalPDFso traininginventory/statusarenotstale.
- Native CPU final controllers canexecutein4parallelchains (4workers each;robustness8workers;28CPUcores), same per-worker numerical defaults. Demo evaluation beginsduringindependentCPUfreshgeneration.
- Metric definitions now explicitly documentgeometric events versusphysicscompletion,worldX/-Yquantityaxes,pathlengthwalkspeedversusnetbackwardspeed,andseparatetableheldreach/contactcriteria. No thresholds changed.
