# 官方 FrankenMotion → G1 统一动作跟踪

后续主线使用官方 FrankenMotion 生成动作，经 GMR 统一重定向及 50 Hz 参考准备，再由一个共享 tracker 在 MuJoCo 中执行。生成端不使用此前训练的 shared20 adapter。在线强化学习代码位于 `shared_motion/rl/`，入口和 Hydra 配置分别位于 `scripts/` 与 `config/tracker_rl/`。

服务器官方生成权重为 `/home/pku/frankenmotion/pretrained/official_20260910/frankenmotion.ckpt`，SHA256 为 `c9dca1988dd08dd9e2164ac4cf6ae8fece23011a6371e83d502cdfabe5352e18`。权重、SMPL 和仿真资产不放入 Git。

当前研究同时评估随机初始化的共享 PPO tracker，以及冻结官方 SONIC mode0 基座、从零初始化残差头的在线 PPO。两条路线均不使用历史专项教师、教师动作数据或蒸馏损失。预定训练与独立测试已完成：从零两个种子普通测试严格成功率为87.5%和82.5%（旧tracker为72.5%）；SONIC+PPO为92.5%（官方SONIC为77.5%）。匹配无偏离终止对照为57.5%。关节误差、失败动作和统计限制见最终报告，不能只看成功率。协议、数据索引、来源审计和复现步骤见 [实验目录](experiments/tracker_rl_20261009/) 与 [复现说明](experiments/tracker_rl_20261009/reproduce.txt)。

## 新 tracker 的 follow 接入

```bash
python scripts/follow_tracker_rl.py reference=PREPARED_REFERENCE.npz \
  policy=EXPORT/policy.pt contract=EXPORT/contract.json \
  scene=EXPORT/scene.mjb initial_state=INITIAL.npz output=NEW_OUTPUT
```

`PREPARED_REFERENCE.npz` 必须是 `prepare_tracker_motion.py` 生成的完整 50 Hz G1 参考，包括 qpos、关节状态及刚体 FK 数组；不能直接传人体动作或原始 GMR NPZ。`INITIAL.npz` 包含完整 MuJoCo `qpos` 和可选 `qvel`，缺少 qvel 时使用零速度。显式提供的状态会原样使用，不会吸附到参考姿态；调用方应保证参考与初始状态的世界坐标对齐。省略 `initial_state` 才使用基准评测的参考起始姿态和速度。

SONIC+PPO 导出需额外传入 `policy_kind=sonic_rl legacy_contract=EXPORT/sonic_contract.json`，并保留与 `policy.pt` 同目录的 `policy.json`（兼容预览布局元数据）。复制模型时应保留完整导出目录。每段开始重置策略历史；输出的 `final_state.npz` 是最后一步后的真实 qpos/qvel，可用于下一段初始化。逐帧 rollout 保存控制前状态，不能把其最后一帧当作最终状态。

原有 `python -m shared_motion.tracker.simulate` 接口也支持新的 schema 2 导出契约，会按契约选择新 tracker，并保存独立的 `*_final_state.npz`。新路径使用统一的参考跟踪终止条件；历史路径仍保持原有行为。外部初始状态接入已验证，但不表示模型具备任意初始姿态恢复能力。

## 历史 shared20 接口与训练记录

以下为原 main 保留的历史接口。其生成器、教师训练入口和权重说明不属于上述官方生成＋无专项教师在线 RL 主线。


本分支保留已选用的 20 类命令生成方案和一个 G1 tracker，并提供可配置的三阶段训练、推理、数据采样和仿真接口。历史试验、按任务选权重的代码、旧 demo、三维 reach 试验均不包含在本分支。原分支和服务器实验目录保留。

## 当前结构

- `shared_motion/adapter/`：55 维命令输入 → 一个共享残差网络 → 冻结的 FrankenMotion 扩散模型。包含命令定义、50 步 DDIM、人体 FK、统一训练和推理入口。
- `shared_motion/tracker/`：一个 SONIC mode0 时序骨干 + 一个共享残差头。单个导出权重输入 2350 维，输出 29 维；每个 episode 开始调用 `reset()`。
- `shared_motion/training/`、`scripts/`、`config/`：可选独立root或统一命令架构的Hydra三阶段训练。
- `shared_motion/data.py`：两阶段均衡采样，由适配器和 tracker 的训练入口共同使用。
- `src/`、`configs/`、`prepare/`：保留 FrankenMotion 上游模型及必要数据工具；不是额外任务适配器或 tracker。
- `configs/release/`：最终权重来源、SHA256、命令定义和 G1 观测/执行约定。

当前权重是 shared20 generator v10 / step2700 和 tracker v3 / step4000。所有任务共用它们；root control 已在同一命令网络内。权重与 SMPL、文本缓存、场景资产不写入 Git。`configs/release/local_assets.json` 是本服务器路径示例，可自行修改。

## 均衡采样

不再为每个任务截取固定 100 条。保留 manifest 中全部 `split: train` 记录：每轮打乱任务顺序，各任务取一条；在该任务内部均匀、有放回采样。任意训练 epoch 中各任务抽取次数最多相差 1。少样本任务可以重复采样，大任务的全部样本都有机会被抽到。验证/测试记录不进入训练。

适配器 manifest 为 JSON 列表，每条包括 `task`、`split`、`prompt_cache`，可含 `command` 和人体动作 `path`。未提供 command 时在该任务范围内随机采样。tracker manifest 每条包括 `task`、`split`、`path`，path 指向含 `observations`、`sonic_actions`、`latents`、`actions` 的训练 NPZ；可含 `complete`。

路径可相对 manifest 文件。不要把已截断的数据清单当作全量数据；重新导出全量训练清单即可使用全部数据。训练总量由 `--steps` / `--batch-size` 控制，和每类数据量解耦。

## 入口

在仓库根目录运行。生成端使用 PyTorch、Hydra/OmegaConf 及上游 `environment.yml` 的依赖；仿真端还需 MuJoCo、SciPy。现有服务器分别可用 `/home/pku/frankenmotion/.conda/bin/python` 与 `/home/pku/frankenmotion/work/mjlab_stable_env/bin/python`。

```bash
python -m shared_motion.adapter.infer --checkpoint GENERATOR.pt --prompt-cache reach.pt --task reach --command 0.4 --seed 0 --output reference.npz --device cuda
python -m shared_motion.adapter.train --initial GENERATOR.pt --manifest train.json --skeleton skeleton.npz --steps 2700 --output generator.pt --device cuda
python -m shared_motion.tracker.train --initial TRACKER.pt --manifest teacher_train.json --steps 4000 --output tracker.pt --device cuda
python -m shared_motion.tracker.simulate --actor TRACKER.pt --scene scene.mjb --contract configs/release/inference_contract.json --reference motion50.npz --initial-state initial.npz --output rollout.npz
python -m unittest discover -s tests -v
```

`adapter.infer` 输出 SMPL-RIFKE 人体参考；GMR 重定向属于外部数据转换步骤。`tracker.simulate` 接收已转换的 50 Hz G1 参考（joint/body pos、quat、vel），以及含完整 MuJoCo `qpos` 的初始状态文件。不要把人体 NPZ 直接传入 tracker。tracker 权重旁必须有同名 `.json`，用于读取 preview offsets。仿真仅初始化一次状态，无任务分流、逐帧状态覆盖或运行时权重混合。

整理后的训练入口用于后续继续训练，不声称从零精确复现历史多阶段试验。已选权重没有被此次整理修改。拍手接触、侧移、跳跃、深蹲及高速慢跑仍有已知质量限制；单纯不摔倒不等于语义或参数执行成功。

## 可配置三阶段训练

新增入口见 [scripts/README.md](scripts/README.md)，Hydra配置在 `config/`。支持独立RootControl＋20类TaskControl与main的55维SharedCommands两种架构，损失可独立切换。默认从配置指定的官方backbone权重初始化，依次执行root条件预训练、真实动作监督、自由生成微调。原20类发布包与tracker入口保留；独立turn/spin训练器和权重未并入；turn数据已修复为经过准入的行进转弯事件。阶段训练需完整20类的motion/text缓存清单，缺失任务会直接报错。
