# 共享任务适配器与统一 Tracker

本分支只保留已选用的 20 类命令生成方案和一个 G1 tracker，以及训练、推理、数据采样和仿真接口。历史试验、按任务选权重的代码、旧 demo、三维 reach 试验均不包含在本分支。原分支和服务器实验目录保留。

## 当前结构

- `shared_motion/adapter/`：55 维命令输入 → 一个共享残差网络 → 冻结的 FrankenMotion 扩散模型。包含命令定义、50 步 DDIM、人体 FK、统一训练和推理入口。
- `shared_motion/tracker/`：一个 SONIC mode0 时序骨干 + 一个共享残差头。单个导出权重输入 2350 维，输出 29 维；每个 episode 开始调用 `reset()`。
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
