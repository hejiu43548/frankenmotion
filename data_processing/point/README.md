# point 合并为 reach XYZ（2026-10-10）

用户确认：“将 point 合并进 reach，统一改为 XYZ 控制”。当前已完成数据与控制接口迁移，**等待数据预览验收，没有训练新模型**。最新完整数据在 Betail `/mnt/sda2/frankenmotion/outputs_amass/source_repair_reach_xyz_20261010_v2`；V1为诊断版，不作当前候选。

| 项目 | train | val |
| --- | --- | --- |
| 旧 point | 115 | 10 |
| 旧标量 reach | 301 | 47 |
| 新 reach XYZ | 21 | 3 |
| 新独立录制 | 19 | 3 |

活动任务变为19类，point不再单独采样。为保护其他任务，原ID保持：reach=1，旧point=14槽位保留但禁用；模型嵌入仍有20个槽位，不等于20个活动任务。其他18类清单和缓存逐条保持原样，包含已通过的jog、march、jump及暂定通过的strike/kick。lean移除仍暂停。

[10个不同TRAIN来源的真实样本预览](http://127.0.0.1:50086/reach_xyz/index.html)。紫圈为目标，紫线为右腕轨迹；样本按HDM优先和XYZ差异选取，不是随机误差率估计。最终10条各6关键帧、双视角已检查，共60个时刻；视频20fps/帧数/哈希已核验，不称所有24条逐帧人工验收。

## 旧数据问题

`audit.py`核验旧point的125个真实源切片。35条caption含“starting point／回到起点”等地点词，旧正则 `\bpoint(?:ing|s)?\b` 没有分辨地点与手指动作；示例包括侧步回原点、走路坐下后回起点、上下台阶回起点。103条裁剪没有可信BABEL point帧标签重叠，这只表示证据不足，不能全部判错误；其中11条来源有point帧事件但裁剪完全错过。43条根移动超过30cm。上述集合有重叠。

现有代码的旧point不是高度指标，而是右腕最大前伸距离；旧reach也是右腕前伸距离的分位数。两者原来都只有一个标量，无法表达三维位置。此次按用户要求统一为右腕目标XYZ，不沿用旧标量标签。

## XYZ定义与实现

- 原点：动作第一帧骨盆；X朝初始身体前方，Y朝初始身体左侧，Z竖直向上；单位米。
- 控制关节：右手腕（SMPL关节21）。不声称控制手指姿态或指向射线。
- 目标：最后5个有效帧（20fps约0.25秒）右腕位置的均值。三个轴来自同一停留窗口，绝不分别取各轴最大值。
- 动作：从低位/近自然初姿伸手到目标，末尾短暂停留。截掉抓取后搬运/回收等后续阶段；不包含必须回到初始姿态的收手命令。
- 语义版本：`right_wrist_xyz_v1`，见 `shared_motion/training/reach.py`。
- 命令编码两种控制器均接收3个分量；旧标量任务在批内表示为 `[value,0,0]`，维度mask保证无效轴不参与归一化损失。with_root输入由任务嵌入+1值改为+3值，without_root当前阶段特征由55变58。原发布类和旧快照仍保持原协议。
- 数据加载拒绝旧标量reach、point活动任务及损坏的XYZ缓存。监督、自由生成损失和验证扫描使用同一XYZ测量，报告三维欧氏位置误差。
- 检查点记录新的任务/命令协议，旧单标量阶段检查点不能直接作为XYZ模型加载。外部已验证root分支仍可按既有范围导入；未重训或修改历史模型。
- reach扫描使用真实验证目标，自由生成阶段使用真实批内reach目标；不从任意XYZ长方体随机采样并假称可达。三个轴的归一化区间分别是[-0.15,0.85]、[-0.8,0.8]、[-0.6,1.1]米，仅为数值尺度，不是已验证的可达空间。

新检查点训练好后，推理形式为 `task=reach 'command=[0.45,-0.20,0.30]'`。当前没有训练这个新模型，所以不宣称已获得三维控制效果。

## 来源、真实裁剪与准入

脚本保存在本目录，阈值在 `config/reach_xyz_repair.yaml`。BABEL point类别直接作为候选；reach/grab/grasp描述需同时有兼容的手臂/手部/抓取/交互类别。文字只作类别细化与排除，不能仅靠整段正则命中准入。帧标签优先；序列回退必须明确单动作。23条异常统一0–1秒时间标签隔离。

从288个候选事件中保留24条；33个test/未分配事件不参与，3个缺raw，228个没有满足当前准入的区间。项目原family划分权威，不因BABEL标签文件名而改train/val。HDM05保留3条，均来自同一个训练录制的不同目标片段；其他21条来自BABEL支持的来源。官方HDM cuts未获得精确对应，不称原生裁剪边界已核验。

在标注内寻找5帧稳定目标，最多向前扩1秒找接近阶段，扩展部分同样排除冲突帧标签；裁15–70帧真实动作，不补帧、不改速度、不修改轨迹。手腕速度和支撑高度等是运动学检查，不是接触真值。

主要准入条件：右腕位移≥16cm、末尾腕部相对目标最大偏差≤6cm/平均速度≤0.35m/s；手臂伸展≥30cm、肘角≥125°；根漂移≤12cm、脚最大位移≤14cm、躯干倾斜≤0.45rad、身体朝向跨度≤20°；左腕位移≤18cm，右腕位移至少为其1.4倍，左腕不能持续高举（最高≤骨盆上25cm）。

预览暴露的误入已增加专项门槛：起始右腕≤骨盆上15cm，肩腕方向离向下垂直≤40°，避免从已伸出姿势收手；目标相对肩水平伸出≥20cm或高于骨盆20cm，避免把垂手恢复姿势当目标；位移/腕路径≥0.60、趋近目标的距离变化比例≥0.85，避免多次摆臂、绕路和返回段。右臂前后变化均保留实际轨迹，低位和侧向目标不简单强制成前上方。

## 数据量和覆盖限制

训练XYZ实际范围：X约0.000–0.603m、Y约-0.611–0.100m、Z约-0.004–0.938m；验证仅3个录制，X约0.334–0.575m、Y约-0.238至-0.128m、Z约0.035–0.663m。逐轴极值的笛卡尔积不是已覆盖空间，侧向、低位、跨身体方向和组合目标仍稀疏；不能宣称支持任意位置泛化。

## 核验与复现

24缓存全部等于真实源切片；独立NumPy XYZ重算最大误差5.97e-8m；CLIP/PCA/局部掩码、19任务加载、family划分、无重复区间、其他18类逐条一致均通过。相关25项测试通过：XYZ梯度/双控制器/混合批/验证扫描/旧协议拒绝、规则回归、阶段单元测试中的合成模型断点重放、turn回归、批量量测与启动器检查。合成测试不是对用户AMASS数据进行训练，没有启动训练实验。

Betail执行副本 `/mnt/sda2/frankenmotion/analysis/point_repair_20261010`。Python `/home/psirobot/projects/frankenmotion/.venv_unified/bin/python`；`PYTHONPATH=/mnt/sda2/frankenmotion/analysis/wave_source_repair_20261009/deps:/tmp/frankenmotion_hydra_deps:.`。

```bash
python -m data_processing.point.audit
python -m data_processing.point.repair output=/path/to/new/output inspect_only=true
python -m data_processing.point.repair output=/path/to/new/output inspect_only=false
python -m data_processing.point.verify output=/path/to/new/output
python -m data_processing.point.review output=/path/to/new/output
python -m unittest data_processing.point.test_rules tests.test_reach_xyz tests.test_staged_training tests.test_walking_turn tests.test_batched_measure tests.test_launch20 -v
python -m data_processing.export_task_audits --data /path/to/new/output --out /path/to/audit --review-state data_processing/review_state.json
```

`reach_index.json`记录原标签、来源、帧边界、XYZ和准入指标；`rejected.json`记录未通过事件及各失败门槛计数；`verification.json`、`provenance.json`、`source_snapshot.json`及 `source/` 保留核验和冻结代码。小报告在 `.codex/reach_xyz_repair_20261010/`；大缓存/视频保存在Betail。旧point/旧reach和V1保留为历史，不自动转换旧模型或更新历史训练结果。
