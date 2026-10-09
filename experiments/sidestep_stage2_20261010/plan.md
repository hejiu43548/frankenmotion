# Sidestep 单任务阶段二试训（2026-10-10）

用户审核HDM05侧步10条训练来源后反馈“问题不大，单任务跑一下side step”。候选暂定通过，本次仅训练sidestep，其他任务继续暂缓。

使用source_repair_sidestep_20261010_v2的34条train、6条val，来源10/2个原始录制。训练方向为右侧镜像，左侧仅作来源参考；没有修改数据、命令范围或划分。

沿用wave/strike/kick试训设置：仅stage2，冻结官方backbone与imported_root，TaskControl重新初始化；10000步、batch8、lr1e-4、seed20261010，每1000步固定10档0.4–1.2m命令/50DDIM扫描。最终10000步模型为主要交付，同时报告固定扫描的最佳步数，不自动延长或进入stage3。

最终额外评估全部6条val条件×3新seed（85101–85103），每个条件/seed扫描10档命令。类别文本相同、长度可能重复，条件哈希用于识别重复，不能将180条生成当180个独立验证来源。视频为最终固定扫描低/中/高3档，无重定时、缩放或地面修正。

入口scripts/run_single_task_stage2.py和scripts/evaluate_single_task_stage2.py，均传--config-name=sidestep_stage2。完整权重与rollout保存在Betail /mnt/sda2/frankenmotion/outputs_amass/sidestep_stage2_20261010；冻结训练器为/mnt/sda2/frankenmotion/code/shared20_walking_43bf05f。
