# Fixed-world reach / strike

本分支基于 main `1ac31e2`，包含官方 FrankenMotion 底座上的 adapter 监督训练、固定世界目标 PPO 残差训练、评估与推理。没有使用旧专项教师或蒸馏权重。PPO 在生成动作上优化关节残差，是单步条件策略优化，不是物理仿真 tracker 的 PPO。

## 入口

- `data_processing/point/prepare_fixed_training.py`：接入清洗后的 reach 数据。
- `scripts/prepare_fixed_targets.py`：固定世界目标数据准备。
- `scripts/train_strike3d.py --config-name train_fixed_target`：reach/strike 共用 adapter 训练入口。
- `scripts/rl_fixed_target.py`：PPO 残差训练。
- `scripts/evaluate_fixed_target.py`：固定目标评估。
- `scripts/infer_fixed_target.py`：给定 XYZ 的生成推理。
- `config/*fixed*.yaml`：对应 Hydra 配置，外部数据和权重路径需按环境覆盖。

当前 reach 使用 21 条训练片段（19 个来源动作族）和 3 条验证片段（3 个来源动作族），均为右手。默认控制点是手腕。目标由初始身体朝向定义：X 前、Y 左；世界 Z 是离地高度。设定后目标在世界中固定。XYZ 可以同时改变。

## 本次视频代码与服务器资产

一次性渲染脚本保存在 `.codex/fixed_target_videos/`，按仓库规范不作为长期 scripts 入口。`axis_generate.py` / `axis_render.py` 是单轴扫描；`joint_generate.py` / `joint_render.py` 是 10 个 XYZ 组合目标。脚本保留实际实验服务器路径，需要在相同环境执行，迁移环境时须修改路径。

服务器 `pku@100.94.2.67`：

- Python：`/home/pku/frankenmotion/.conda/bin/python`
- 实验根目录：`/home/pku/frankenmotion/work/fixed_target_20261010`
- 官方底座：`/home/pku/frankenmotion/pretrained/official_20260910/frankenmotion.ckpt`
- reach adapter：实验目录下 `reach_sft_seed84001/final.pt`
- reach PPO：实验目录下 `reach_ppo_seed86001/final.pt`
- 数据：实验目录下 `reach_data/`
- 组合视频和结果：实验目录下 `xyz_joint/`
- 单轴视频和结果：实验目录下 `position_sweep/`

先执行对应 generate，再执行 render。两组视频均使用相同文本、seed=51000、39 帧动作，结尾加 20 帧显式标注的定格。视频中的误差是最后 5 帧平均手腕位置到目标的距离。十个组合位置中七个误差小于 10 cm，另三个为 10.4、14.4、19.6 cm；不代表整个空间均可达。

大权重、数据缓存和视频不加入 Git。完整训练复现包已交付为 `fixed_target_reach_strike_bundle.zip`，包含权重、缓存、复现命令、来源校验及报告。此分支还保留先前 strike 实验入口；本次使用的是上述 fixed_target 入口。
