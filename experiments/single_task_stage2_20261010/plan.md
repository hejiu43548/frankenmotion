# 单任务第二阶段命令响应试验（2026-10-10）

用户授权分别训练 wave、strike、kick，仅第二阶段。三个独立 TaskControl 从相同随机初始化出发，使用同一官方冻结 backbone 和已经导入的冻结 root；不使用旧阶段二或阶段三的 TaskControl。训练器为冻结代码快照 `43bf05f`，本次未改模型或损失。

每个任务 10000 步，batch 8，学习率 1e-4；每 1000 步对固定验证文本及相同噪声做 10 档 50 DDIM 扫描，保留初始化、所有扫描、检查点和实际数据索引。每任务约 80000 次样本呈现，独立来源远少于这个数字。最终结果以 10000 步为主，同时报告扫描选择的最佳步数，避免只展示最好结果。额外使用全部验证文本和三个未用于训练扫描的新噪声种子评估命令响应；这是重复验证，不是独立测试集。

数据使用 source_repair_kick_20261009_v1 的逐任务过滤列表：wave 45/3，strike 13/1，kick 33/2（train/val）。kick 试训使用候选数据，不代表用户已经完成语义复核。保持原命令范围，不缩放动作或改标签填空档。范围覆盖、混合动作残留及来源偏差均作为已知局限。

入口：scripts/run_single_task_stage2.py；配置：config/single_task_stage2.yaml；评估与视频：scripts/evaluate_single_task_stage2.py。大型权重与 rollout 位于 Betail `/mnt/sda2/frankenmotion/outputs_amass/single_task_stage2_20261010`，小型报告和审计索引同步到本目录。
