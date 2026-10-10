# jog直线跑步修复 V3（2026-10-10）

用户指出上一版混入旋转/弧线跑步，要求筛出直跑。上一版只检查身体相对前向运动，确实允许沿弧线跑；本轮增加世界坐标路径和全程朝向检查。仅修改jog，march及其余18类逐条保留。

当前数据：Betail `/mnt/sda2/frankenmotion/outputs_amass/source_repair_jog_straight_20261010_v3`。

| 项目 | train | val |
| --- | --- | --- |
| jog样本 | 9 | 2 |
| jog独立录制 | 8 | 2 |
| 原命令0.8–2m/s范围内 | 6 | 2 |
| march（完全不变） | 29 | 1 |

[全部9条jog训练样本](http://127.0.0.1:50086/jog_straight/jog/index.html)。不足10条时全部展示，不重复凑数；其中2条来自同一HDM录制的不同时间段。用户已于2026-10-10通过jog和march，明确保留数据量少的限制。参数覆盖缺口继续记录，没有训练。

## 筛选与裁剪

在V2已通过BABEL类别和跑步步态核验的97条jog窗口内，搜索长度至少40帧（2秒）的连续直跑子段，优先最长窗口，再选剩余不重叠段。没有重新全面挖掘AMASS全部原录制，因此本次11条不是整个AMASS直跑库存上限。6条旧窗口整段保留，另5条裁掉转向部分后保留；86条无满足当前时长、步态及直线门槛的子段。

所有原跑步检查继续执行；新增配置 `config/jog_straight_repair.yaml`：

- 首尾净位移 / 全程路径长度 ≥ 0.985。
- 全程根位置偏离首尾连线的最大距离 ≤ 0.12m。
- 7帧平滑后的身体朝向全程跨度 ≤ 20°，首尾变化 ≤ 10°。
- 平滑根速度方向全程跨度 ≤ 20°。

检查全程跨度，不能只用首尾角差，否则S弯、转出去再转回来会漏检。角度unwrap后检查，保证跨±180°不会误判；任意世界旋转和平移不改变决定。轻微步态摆动允许，明显弧线和转身排除。所有轨迹是原动作真实裁剪，无矫直、缩放或重定时。

保留的11条中7条来自HDM05（6train/1val），另有ACCAD、CMU、BMLmovi、BioMotion各1条。ACCAD为 `Male2Running_c3d/C3 - run_poses` 的0–2.45秒；同目录内转向动作不能自动当直跑。HDM是录制中的合格直段，不声称整份录制为直线跑步，更不声称核验了官方HDM原生cuts。

当前jog语义版本为 `straight_running_v2`，准备入口拒绝导入旧的 `travelling_running_v1`。任务ID、速度量和命令范围未变。训练实际速度1.170–3.635m/s，范围内仅6条、10档中5档为空；验证仅2个录制，不足以证明覆盖充分或泛化。

## 核验与审计

20项测试通过，新增直跑接受、身体沿弧线同步转向拒绝、S弯拒绝、路径直但身体转向拒绝、世界旋转/平移不变性。41缓存（11jog+30未变march）源切片、文本/PCA/掩码、量独立重算、全20任务划分隔离及其他19类逐条一致均通过，最大量误差1.79e-7。

9条训练视频20fps、帧数及同步文件SHA256检查；本轮检查全部9条各6个关键帧、双视角，共54个时刻。不是全量逐帧人工验收；完整视频已交用户复核并通过。march预览随总览提供，其数据不变，沿用上一轮检查。

`straight_audit.json` 按97个父窗口记录原直线指标、失败原因、获选子段相对帧区间、搜索中失败规则计数。`running_index.json` 有原始来源、parent_key/cache、最终绝对帧边界、步态和直线指标；`verification.json`记录全量验证；`provenance.json`及 `source/` 保存输入哈希与代码快照。旧V2的类别审计与原始来源血缘继续保留。所有样本数均为窗口数，不能当独立来源数。

## 复现

当前Betail执行副本 `/mnt/sda2/frankenmotion/analysis/jog_straight_repair_20261010`。使用项目 `.venv_unified/bin/python` 和上一轮相同依赖/PYTHONPATH，选择新的输出目录，避免覆盖已完成版本：

```bash
python -m unittest data_processing.jog_march.test_straight data_processing.jog_march.test_rules tests.test_batched_measure -v
python -m data_processing.jog_march.straight output=/path/to/new/output inspect_only=true
python -m data_processing.jog_march.straight output=/path/to/new/output inspect_only=false
python -m data_processing.jog_march.verify --config-name jog_straight_repair output=/path/to/new/output
python -m data_processing.jog_march.review --config-name jog_straight_repair output=/path/to/new/output
python -m data_processing.export_task_audits --data /path/to/new/output --out /path/to/audit --review-state data_processing/review_state.json
```

这些命令不会启动训练。其他19类、先前jump/strike/kick决定及历史模型保持不变。
