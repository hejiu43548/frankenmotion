# 代码风格与重构规范

本规范适用于本仓库新增和修改的代码，尤其是 `shared_motion/`。修改现有代码时，应遵守以下要求，避免重新引入含义不明的命名和压缩式排版。

## 任务关键代码路径

以下路径均相对仓库根目录。优先根据任务定位到对应入口和实现；三阶段训练与原发布版适配器使用不同的配置和检查点协议。

### 三阶段训练与推理

| 路径 | 职责 |
| --- | --- |
| `scripts/README.md` | 三阶段训练操作说明、数据格式、两种控制器、恢复与推理约定。 |
| `scripts/train.py`、`scripts/infer.py` | Hydra 训练与阶段检查点推理入口。 |
| `scripts/train_stage1.sh`、`scripts/train_stage2.sh`、`scripts/train_stage3.sh` | 各训练阶段的启动脚本。 |
| `scripts/run_stages.py` | 阶段二、三的顺序调度、状态记录和阶段衔接检查。 |
| `shared_motion/training/runner.py` | 训练循环、验证扫描、检查点保存与恢复、实验协议及来源校验。 |
| `shared_motion/training/model.py` | 官方底座加载、控制器构建、条件编码和生成采样。 |
| `shared_motion/training/adapters.py` | 独立 `RootControl` 与 `TaskControl` 的网络实现。 |
| `shared_motion/training/losses.py` | root、监督训练、自由生成及 replay 等可配置损失。 |
| `shared_motion/training/catalog.py`、`shared_motion/training/geometry.py` | 20 类任务的命令范围、批量指标计算与训练使用的骨架 FK。 |
| `config/train.yaml`、`config/infer.yaml`、`config/pipeline.yaml` | 三阶段训练、推理和流水线的顶层 Hydra 配置。 |
| `config/controller/`、`config/stage/`、`config/loss/`、`config/backbone/` | 控制器选择、阶段配方、损失及官方底座配置。 |

### 数据准备与任务语义

| 路径 | 职责 |
| --- | --- |
| `shared_motion/training/data.py` | 阶段训练缓存读取、任务覆盖及数据划分检查、可恢复采样器。 |
| `shared_motion/training/turn.py`、`shared_motion/training/turn_data.py` | `walking_turn_v2` 的有符号转角、命令采样、事件准入与来源验证。 |
| `scripts/prepare_amass20.py`、`config/prepare_amass20.yaml` | 完整 20 类动作缓存和数据清单准备。 |
| `scripts/repair_turn_data.py`、`config/repair_turn_data.yaml` | 用经过验证的行进转弯事件替换清单中的 turn 数据。 |
| `config/data/amass20.yaml`、`config/data/turn_sources/walking_turn_v2.yaml` | 数据路径、任务列表及 turn 来源约定。 |
| `scripts/import_root.py`、`config/import_root.yaml` | 校验并导入外部 root 分支。 |
| `scripts/run_single_task_stage2.py`、`scripts/evaluate_single_task_stage2.py`、`config/single_task_stage2.yaml` | 单任务阶段二实验的启动与评估。 |

### 发布版适配器与 G1 tracker

| 路径 | 职责 |
| --- | --- |
| `shared_motion/adapter/network.py` | 55 维 `SharedCommands` 及 `UnifiedControl` 残差注入；阶段训练的统一命令架构也复用此实现。 |
| `shared_motion/adapter/schema.py`、`shared_motion/adapter/catalog.py`、`shared_motion/adapter/inputs.py` | 原发布协议的任务定义、命令字段、范围及输入编码。 |
| `shared_motion/adapter/model.py`、`shared_motion/adapter/sampling.py` | 发布版检查点加载、命令构造与 DDIM 采样。 |
| `shared_motion/adapter/kinematics.py` | 发布版人体 FK 和原任务指标；与阶段训练的骨架高度及 turn 语义需区分。 |
| `shared_motion/adapter/infer.py`、`shared_motion/adapter/train.py` | 原发布版适配器的推理与续训模块入口。 |
| `shared_motion/data.py` | 发布版适配器和 tracker 共用的 manifest 读取与任务均衡采样。 |
| `shared_motion/tracker/model.py`、`shared_motion/tracker/runtime.py` | G1 tracker 网络、历史状态、导出模型加载及推理。 |
| `shared_motion/tracker/train.py`、`shared_motion/tracker/simulate.py` | Tracker 训练和 MuJoCo 仿真入口。仿真输入为重定向后的 G1 参考动作。 |
| `configs/release/` | 发布权重来源、命令协议以及 tracker 观测和执行约定。 |

### 底座、验证与实验输出

- `src/model/backbones/frankenmotion.py`、`src/model/gaussian.py`、`src/tools/geometry.py`：上游去噪网络、扩散实现和旋转几何工具。
- `tests/test_staged_training.py`、`tests/test_launch20.py`：阶段训练、控制器及损失组合、完整任务流水线的验证。
- `tests/test_batched_measure.py`、`tests/test_walking_turn.py`、`tests/test_sampling.py`：批量指标及梯度、行进转弯语义、任务均衡采样的验证。
- `experiments/`：按“实验名_日期”组织报告、配置快照和数据索引；大权重、动作缓存和视频使用配置指定的外部资产目录。

注意：`config/` 是当前阶段训练的 Hydra 配置；`configs/` 保留上游配置和发布协议，修改时不要混淆。完整测试入口为 `python -m unittest discover -s tests -v`。

## 命名必须表达含义

- 禁止使用含义不明的单字母变量、参数或属性，例如 `a`、`b`、`c`、`p`、`q`、`x`、`y`、`z`。循环下标和推导式变量也必须有明确名称。
- 禁止使用难以理解的缩写，例如 `pp`、`tp`、`cc`、`tr`、`so`。应根据用途命名为 `predicted_positions`、`target_positions`、`replay_commands`、`teacher_residuals` 等。
- 允许 `_` 表示确实不使用的值，以及 `self`、`cls`、`np`、`nn` 等通用约定。领域术语如 `yaw`、`pose` 可以保留。外部接口、文件格式或 checkpoint 要求的固定名称应保持兼容，并在含义不直观时添加说明；这些例外不能作为内部变量随意缩写的理由。

## 使用正常、易读的排版

- 使用四个空格缩进；禁止通过分号连接多条语句，或将控制流语句及其主体压在同一行。
- 每条 `import` 语句只导入一个模块，不写 `import argparse, json, torch`。同一模块的 `from ... import ...` 可以导入多个名称。
- 长表达式、函数调用和参数列表使用括号自然换行。不得为压缩行数牺牲可读性，也不得为了缩短行宽退回无意义的变量名。
- Python 代码使用 Black 默认格式。注释应解释单位、形状、坐标约定或设计原因，并与实际命名保持一致。

## Best Practice
- 将启动入口放置在scripts
- 采用hydra管理配置
- 将试验结果放入 experiments/实验名_日期，应包含报告，复现所需的配置副本和数据索引文件，不要包含大视频或数据
- commit格式采用
```
feat|doc|refactor|fix|chore(scope): digest

detail
```
格式
- 只有具有长期使用意义的脚本才放置到scripts文件夹，否则应该放到.codex的临时工作区，实验记录也是同理，除了用户指定的大规模重新训练外，其余
小型的实验不需要记录到experiments
- 复现数据筛选的代码需要保存到data_processing中，方便复现和代码写作