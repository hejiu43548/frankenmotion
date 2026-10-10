# 当前：jog改为直线跑步（V3）

用户已于2026-10-10通过当前jog V3（9train/2val）和march（29train/1val），原话“jog和march都可以通过了，除了数据量少都没问题”。小样本及参数覆盖限制保留，未训练。此决定覆盖下方历史“待复核”状态；旧版本jog不因此通过。

用户进一步要求排除旋转/弧线跑步。当前jog为9train/2val，march29/1逐条不变，其他18类也不变。最新完整数据为 `source_repair_jog_straight_20261010_v3`，见 [直跑复核与复现](STRAIGHT_REVIEW.md)。下文是上一轮V2联合修复记录，其中允许弧线跑步的jog语义已经被本次要求覆盖。未训练。

# jog / march 联合修复（2026-10-10）

用户要求：正常移动跑步进入 jog，原地跑步进入 march，两类同步修复。当前候选已生成并核验，等待用户复核；没有启动训练。

| 任务 | 原 train / val | 当前 train / val | 当前独立录制 train / val | HDM05 样本 |
| --- | --- | --- | --- | --- |
| jog | 742 / 92 | 76 / 21 | 36 / 9 | 36 |
| march | 42 / 2 | 29 / 1 | 28 / 1 | 2 |

完整数据：Betail `/mnt/sda2/frankenmotion/outputs_amass/source_repair_jog_march_20261010_v2`。以 jump V3 为底，仅替换 jog/march，其余18类记录逐条相同。原始数据和历史版本保留。V1 是诊断版，V2 修正站立边缘裁剪和步态相位偏差，是当前候选。

[两个任务视频总览](http://127.0.0.1:50086/jog_march/index.html) · [jog 10条](http://127.0.0.1:50086/jog_march/jog/index.html) · [march 10条](http://127.0.0.1:50086/jog_march/march/index.html)

## 语义与来源

- jog：双腿交替、连续向前移动的跑步，允许沿弧线跑。按用户“正常跑步”保留真实较快跑步，不以旧慢跑命令上限删除或改速。
- march：原地交替跑步。站立、原地走/踏步、双脚跳、低速移动不因根位移小而进入此类。
- BABEL必须包含 `run`/`jog` 的 `act_cat`，并检查时间重叠处的其他类别和描述。正则只做排除，不能独立产生准入证据。
- HDM05来源优先，当前38条有BABEL及运动学证据；官方cuts访问受403限制，未获得与现有AMASS文件的精确原生边界映射，不能称已核验原生剪辑类别。
- 68条来自帧级标签（jog60/march8），59条来自明确 `mul_act=false` 且所有标签兼容的单动作序列（jog37/march22）；后者边界是运动补标，不是原生时间标注。
- 1596个候选事件经过检查。两个来源中的23条异常统一0–1秒标签隔离于 `uncertain_time_labels.json`，不参与精确边界推断。候选可能产生多个窗口，因此拒绝计数不能与候选数简单相加。
- 原项目的source-family划分权威；BABEL的train/val文件只是标签来源。不移动原test或未分配来源，不让同一录制跨train/val。新march的23条记录来自旧jog来源，但经过重裁剪，不是直接换标签。

## 实际裁剪与准入

完整阈值见 `config/jog_march_repair.yaml`，实现在 `rules.py` / `repair.py`。

1. 从实际连续速度/方向段提出40–100帧（20fps，2–5秒）的候选；识别实际脚活动，将站立前后段裁掉，再完整复检。脚活动门槛4cm，两端允许1帧支撑边缘。裁剪后少于40帧拒绝，不补帧或重定时。
2. 每脚至少2个清晰摆动峰，峰高≥7cm、显著度≥5.5cm，交替比例≥80%，同时峰比例≤25%，步频2–5Hz，最长峰间隔≤0.75秒，同脚周期变异系数≤0.45。步频取相邻峰间隔中位数倒数，避免短窗口从不同相位截取导致计数偏差。
3. 重复共同离地代理≥2次且占比≥2.5%，共同支撑连续不超过5帧，排除站立、走路和同步双脚跳。接触/腾空由踝/脚趾高度估计，是运动学代理，不能称接触真值。
4. 支撑候选高度≤3.5cm、竖直速度≤0.30m/s，支撑脚水平速度中位数≤0.35m/s，排除明显滑动和跑步机。文本/上下文同时排除跑步机、上下台阶/坡面、倒跑/侧跑、假跑、舞蹈、跛行、持物、击打及其他混合动作。
5. jog全段平均根速度≥0.65m/s，身体相对前向比例≥80%，侧向比例≤35%，后向比例≤5%，低速停顿比例≤10%。
6. march平均根速度≤0.25m/s、全程根水平偏移≤20cm、终点位移≤15cm。最终最大偏移18.41cm。中间不明确的移动模式直接拒绝，不强行分到march。
7. 躯干倾斜≤0.65rad；同一来源的重复或跨任务重叠片段不能重复纳入。所有数据保留精确源切片、原始标签和裁剪理由。

当前20任务准备入口已禁止从旧caption正则重建这两类；`scripts/prepare_amass20.py` 必须从 `running_data` 导入具有新语义版本、正确split和匹配缓存SHA256的记录。原始证据保留，训练用caption改为准确的移动跑步/原地跑步描述，并重新编码文本与PCA。

## march指标更新与覆盖限制

当前阶段训练的 `shared_motion/training/catalog.py` 中，march仍为双脚踝平均最大抬升高度，基准由首帧改为各脚自身5%分位支撑高度。原地跑步可以从摆腿相位开始，首帧基准会低估甚至近乎消掉高度；新定义不依赖首帧相位或整体竖直偏移。语义版本 `stationary_running_v1`；jog为 `travelling_running_v1`。任务ID17/18不变，原发布协议与历史冻结代码不变，旧检查点不能混用新march量后直接比较。

命令范围暂未修改，也没有为补覆盖而合成、缩放或重定时：

| 任务 | 现命令范围 | 实际train量范围 | 范围内train / val | 缺口 |
| --- | --- | --- | --- | --- |
| jog | 0.8–2m/s | 1.072–3.630m/s | 65 / 20 | 训练0.8–1.04m/s缺覆盖 |
| march | 0.06–0.22m | 0.148–0.478m | 5 / 0 | 仅1条val，其量0.221502m；旧范围与现数据不匹配 |

march的低样本量和验证缺口必须在后续训练前处理或明确接受，不能把本次语义清理视为监督范围充分、泛化已验证。

## 验证与预览范围

- 127个缓存逐一等于真实来源切片；1340个输入哈希核验；独立量重算最大误差5.37e-7。
- 全量重跑准入检查，文本/PCA/局部掩码、有限值、真实MotionDataset加载均通过。其他18类不变；全20类family划分隔离；两类无重复/跨类重叠裁剪。
- 15项测试通过，包含走路、站立、倒跑、同步跳、跑步机、低速移动拒绝，原地/移动跑接受，站立边缘裁剪、相位与竖直偏移不变性、批量变长与梯度、禁止旧清单导入及缓存篡改。
- 每类10个不同TRAIN录制，先选最多2个HDM来源，再按控制量分层；分层余下样本仍可能来自HDM。展示用于核实语义和幅度，不是随机误差率估计。
- 20段视频20fps、帧数和SHA256核验；检查每段6个关键帧、双视角，共120个时刻。未声称全部127段逐帧人工验收。状态仍为 `awaiting_user_review`。

## 复现

在包含本项目依赖、模型资产和BABEL/AMASS的Betail环境，从代码根执行。当前执行副本位于 `/mnt/sda2/frankenmotion/analysis/jog_march_repair_20261010`；Python为 `/home/psirobot/projects/frankenmotion/.venv_unified/bin/python`，辅助依赖需加入 `PYTHONPATH`：`/mnt/sda2/frankenmotion/analysis/wave_source_repair_20261009/deps:/tmp/frankenmotion_hydra_deps:.`。

```bash
python -m unittest data_processing.jog_march.test_rules tests.test_batched_measure -v
# 对新输出目录先只检查；没有生成训练缓存或启动训练。
python -m data_processing.jog_march.repair output=/path/to/new/output inspect_only=true
# 同目录生成最终数据缓存。已存在最终running_index时拒绝覆盖。
python -m data_processing.jog_march.repair output=/path/to/new/output inspect_only=false
python -m data_processing.jog_march.verify output=/path/to/new/output
python -m data_processing.jog_march.review output=/path/to/new/output
python -m data_processing.export_task_audits --data /path/to/new/output --out /path/to/audit --review-state data_processing/review_state.json
```

输出包括 `running_index.json`、全20任务train/val、`rejected.json`、`uncertain_time_labels.json`、`provenance.json`、`source_snapshot.json`及冻结源码、`verification.json`、文本嵌入与缓存、review视频及逐条manifest。`ready_for_training` 仅表示缓存格式可读取，不表示用户验收或训练授权。小报告保存在 `.codex/jog_march_repair_20261010/result/`；大型缓存和视频在Betail。
