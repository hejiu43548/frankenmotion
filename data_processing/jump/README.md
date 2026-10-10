# 原地平地双脚跳来源清理（V3）

当前用户定义：**平地双脚起跳，落回起跳位置**。`jump left/right` 以及向前、向后等方向跳不能进入。这覆盖V2中允许水平位移的解释。不训练、不通过修改轨迹统一落点。

最终候选为 **49 train / 8 val**，来自 **47 / 8 个独立录制**；V2为118/16，原始库存为759/94。其余19类逐条保持不变，lean仍保留。当前状态：**用户审核通过（2026-10-10）**；未授权训练。

- 数据：Betail `/mnt/sda2/frankenmotion/outputs_amass/source_repair_jump_20261010_v3`
- [10条训练视频](http://127.0.0.1:50086/jump/index.html?v=3)，文件名带v3，避免沿用旧视频缓存。
- [逐条来源列表](http://127.0.0.1:50086/audits_jump_v3/jump/index.html)

## 标注准入

候选必须有BABEL `act_cat=jump/hop`，优先帧标签。没有帧标签时，仅接受 `mul_act=false` 且全部标签兼容的单动作序列，并根据实际腾空/支撑补标边界。最终11条来自帧标签、46条来自单动作序列；后者不声称是官方精确事件边界。使用项目原family split；test与未分配来源不准入。

同时检查proc/raw原文及其他标注旁证，排除left/right/forward/backward、sideways、side-to-side等方向描述；移除forward/backwards/sideways movement类别。拒绝单脚、绳跳、开合跳、上下台阶/平台、障碍跳、踢跳、旋转跳及不兼容动作重叠。文字规则只细化已有类别候选，不能单独用于纳入数据。

优先检查HDM05，但官方cuts仍不可访问；可用候选没有通过最终准入，当前HDM05为0。没有为增加数量放宽规则或冒称官方hopBothLegs边界。

## 运动裁剪与落点检查

规则集中在 `config/jump_repair.yaml`，单位沿用训练骨架高度归一化；20fps。双脚支撑高度来自脚踝/足部点估计，不是接触传感器真值。

- 必须包含一次完整双脚腾空：共同离地高度>6cm，连续至少3帧、不超过24帧。前后各至少3帧双脚支撑；双脚离地与落地时间差各≤2帧。
- 前后每脚地面高度变化≤3.5cm，左右支撑高度差≤5.5cm；同步排除台阶/平台标注。
- 搜索前后最多1.5s上下文；起点膝角均值≥145度，末尾≥135度，避免从深蹲计算跳高或截断落地恢复。
- 以起始支撑脚的位置为统一基准，比较起跳前、**首次落地**和最终恢复：脚中心水平偏移≤8cm，每脚≤10cm。另直接比较起跳与首次落地。跳出去后走回来不能通过。
- 根终点偏移≤10cm，全程根水平偏移≤15cm，防止端点相同却中途横移。容差用于真实动捕的小幅漂移，并非把落点变换成完全相同的坐标。
- 控制转角、躯干倾斜、脚间距变化、前后错脚及空中左右脚高度差。根高度峰值在共同腾空期间，根增高≥8cm。
- 裁剪24–100个真实帧，动作数组与来源切片逐元素相同；无补帧、重定时、缩放、镜像、落点平移或轨迹修正。同一物理腾空去重。

标签仍为 `max(root_z)-root_z[0]`，实际重新计算。文本条件改为明确的原地双脚跳并落回同一位置；原标签、时间区间、来源ID、拒绝原因全部保留。使用既有CLIP/PCA编码，action/左右腿启用local mask。

## 结果和核验

57缓存精确源切片、哈希、准入条件、文本/PCA/mask、全部20类family划分通过；其他19类记录完全一致。独立从motion高度维计算跳高，最大误差1.70e-8m。真实MotionDataset加载49/8成功。11个规则回归测试覆盖方向文本、单脚、错时起落、不同地面高度、开合、深蹲基准、横移、跳出后回归及小幅容差。

最终最大脚中心偏移 **7.571cm**、单脚偏移 **9.162cm**，根水平偏移最大11.096cm。训练实际跳高0.130–0.465m，26/49在0.25–0.55m命令范围内；验证0.171–0.352m，5/8在范围内。训练0.40–0.43m、0.49–0.55m仍缺覆盖；没有修改幅度来补齐分布。

10视频来自10个不同TRAIN录制，按跳高分层选择；50个准备/起跳/峰值/落地/结束关键帧已检查。视频20fps、帧数与哈希核验。用户已批准本版数据；不把抽样复核扩大成逐条人工验收，也不把自动支撑代理当作接触真值。

## 复现与审计脚本

所有任务脚本保存在本目录：`audit_original.py`（旧853条审计）、`rules.py`、`repair.py`、`verify.py`、`review.py`、`test_rules.py`。V1/V2保留历史证据，不用于当前数据。

在仓库根目录执行；Betail已备好的依赖如下：

```bash
export PYTHONPATH=/mnt/sda2/frankenmotion/analysis/wave_source_repair_20261009/deps:/tmp/frankenmotion_hydra_deps:.
PYTHON=/home/psirobot/projects/frankenmotion/.venv_unified/bin/python
"$PYTHON" -m unittest data_processing.jump.test_rules -v
"$PYTHON" -m data_processing.jump.repair output=/absolute/new_output inspect_only=true
"$PYTHON" -m data_processing.jump.repair output=/absolute/new_output inspect_only=false
"$PYTHON" -m data_processing.jump.verify output=/absolute/new_output
"$PYTHON" -m data_processing.jump.review output=/absolute/new_output
```

使用新目录复现，已完成输出不会被静默覆盖。最终目录包含jump_index、全任务train/val、rejected、provenance、verification、source代码快照和review视频。运行记录在本地 `.codex/jump_repair_20261010/v3/`，未启动训练。
