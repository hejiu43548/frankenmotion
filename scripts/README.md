# Hydra 三阶段训练

新增训练默认覆盖 main 的20类任务；没有并入独立 turn/spin 试验。旧 `shared_motion.adapter.infer` 和已发布 shared20 权重保持兼容。上游配置继续在 `configs/`；本次阶段训练配置按要求放在 **`config/`**。

使用 Python3.9–3.12、仓库 `environment.yml` 的 PyTorch/Hydra/OmegaConf 依赖。下面命令从仓库根目录运行，`PYTHON` 可指定解释器。三个 shell 入口会先切到仓库根目录。

## 数据和权重

```bash
export PYTHON=/path/to/python
export FRANKENMOTION_CHECKPOINT=/path/to/frankenmotion.ckpt
export FRANKENMOTION_ARTIFACTS=/path/to/large_artifacts

DATA_ARGS=(
  data.train_manifest=/path/to/train.json
  data.val_manifest=/path/to/val.json
  data.path_root=/path/to/cache_path_root
  data.skeleton=/path/to/skeleton.npz
)
```

`config/backbone/official.yaml` 用 Hydra 实例化底座结构和扩散 schedule，并严格加载官方 checkpoint 的 denoiser 与内置归一化统计。默认验证已下载公开权重的 SHA256。换权重时必须明确修改 `backbone.checkpoint` 和 `backbone.expected_sha256`；有意识地省略哈希校验才使用 `backbone.expected_sha256=null`。

训练 manifest 每条需要：`split`（train/val）、`task`、`family`、`key`、`cache`；可含 `target_frames`。`cache` 指向 NPZ，字段为：

- `motion`: T×205 SMPL-RIFKE；`local`: T×408 文本特征。
- `local_mask`: T×408；`tx`: 512；`quantity`: 此片段真实命令量的标量。
- 3≤T≤120；特征与官方模型采用相同归一化/PCA约定。只在 batch 内补齐到120帧并按长度屏蔽。

相对 cache 路径按 `data.path_root` 解释；不随 Hydra 输出目录改变。train/val 按 `family` 隔离。入口检查配置中的每个任务都有数据；已有11类缓存不能直接宣称是20类。main 旧的 prompt-only manifest 不足以执行阶段一/二，需要先准备真实 motion 与 quantity；此改动不编造或复制缺失任务。部分动作指标仍沿用原固定时间窗口，数据准备应保证该任务事件落在有效窗口中。

## 两种控制器

| Hydra 选择 | 架构及阶段含义 |
|---|---|
| `controller=with_root`（默认） | 独立 RootControl＋TaskControl。阶段一训练root，二/三冻结root，只训task。Root结构保持593536参数；任务embedding从11扩展20，Task597888参数。没有新增时间相位或205维输出头。 |
| `controller=without_root` | main的55维 SharedCommands，默认width1024、8664269参数，保留时间特征和205维输出残差。**无独立root分支，不是删除root输入字段**。阶段一用root_profile训练统一网络；阶段二/三继续训练同一个网络，root能力不会被单独冻结。 |

两种配置都从冻结官方底座开始；新控制器的最终残差层归零初始化。只有 walk/back_walk/turn 的生成请求显式携带root值；新增9类沿用main条件定义。turn仍是main的零速度、正向0.45–1.5rad语义，本次没有导入独立walking-turn/spin的任务定义或权重。

## 三阶段启动

给每套控制器不同的 `experiment`。同一次实验固定 `run_date`，避免跨日运行时自动路径变化。

```bash
bash scripts/train_stage1.sh experiment=charlie20 run_date=20261009 controller=with_root "${DATA_ARGS[@]}"
bash scripts/train_stage2.sh experiment=charlie20 run_date=20261009 controller=with_root "${DATA_ARGS[@]}"
bash scripts/train_stage3.sh experiment=charlie20 run_date=20261009 controller=with_root "${DATA_ARGS[@]}"
```

阶段二/三默认读取该实验前一阶段的 `best.pt`，也可以显式指定 `initial=/absolute/path/best.pt`。检查架构、任务列表、底座哈希及底座结构/schedule配置，不接受不同控制器或不同阶段权重。

| 阶段配置 | 默认 recipe |
|---|---|
| `stage1` | 按family/key去重保留最长片段；最多100epoch，patience15，改善阈值1e-5；batch16，accumulate4；AdamW1e-4，wd.01，clip1，speed/yaw独立.15条件丢弃。 |
| `stage2` | 70000步，batch40（每20类循环两次），AdamW1e-4，wd1e-6，clip2；每10000步评估。 |
| `stage3` | 8800步，batch20，AdamW3e-5，wd.01，clip1；10步可微DDIM训练，每1100步评估。 |
| `stage3_main` | 可选main续训配方：2700步、batch1、wd1e-6、50步可微DDIM、每900步评估，默认 `main_replay` 损失。 |

任务类内有放回采样，按随机任务循环保持任务计数均衡。阶段一遍历全部记录；阶段二完成时也检查全部记录至少使用一次。步数过短时会报出覆盖不足，而非伪称完整训练。

```bash
# SharedCommands 的三阶段；第三阶段选择 main-style 默认优化器与损失。
bash scripts/train_stage1.sh experiment=shared20 run_date=20261009 controller=without_root "${DATA_ARGS[@]}"
bash scripts/train_stage2.sh experiment=shared20 run_date=20261009 controller=without_root "${DATA_ARGS[@]}"
bash scripts/train_stage3.sh stage=stage3_main experiment=shared20 run_date=20261009 controller=without_root "${DATA_ARGS[@]}"
```

## 损失切换

损失通过 `config/loss/` 注册，独立于控制器选择；stage/loss类型不兼容时直接报错。

| 选择 | 计算 |
|---|---|
| `loss=root` | 根特征加权重建＋.1speed MSE＋.1yaw MSE。 |
| `loss=supervised` | 加权重建＋.2命令Huber/span＋.1speed MSE＋.1yaw MSE；turn用周期误差。 |
| `loss=free_generation` | 命令Huber/span＋2相对姿态＋.002速度过量＋.01支撑脚速度。固定初始参考只接root条件；SharedCommands参考是阶段三开始时冻结的统一网络，关闭task输入，不声称有独立冻结root模块。 |
| `loss=main_replay` | 命令MSE＋2相对姿态＋.2root位置匹配＋.05速度匹配＋300控制器输出保持；完整冻结初始模型作为teacher，20类命令replay，额外静止/站姿以及clap/arm_circle真实动作约束。 |
| `loss=command_only` | 仅命令Huber，供损失消融。 |

`main_replay` 对有root架构也可用：保持的是TaskControl的四层残差，因该架构无205维输出头而不存在输出头保持项。数据源来自本次缓存接口，支持masked batch，clap/arm_circle直接使用同时间窗缓存；这是main损失的可切换实现，**不是历史v10完整训练的精确复现**。50步反传显存明显更高，优先用 `stage=stage3_main` 的batch1。

```bash
bash scripts/train_stage3.sh controller=without_root loss=main_replay stage.batch_size=1 stage.weight_decay=1e-6 "${DATA_ARGS[@]}"
bash scripts/train_stage3.sh loss=free_generation loss.command_kind=mse loss.support_weight=0.05 "${DATA_ARGS[@]}"
# 仅查看组合配置，不加载数据/权重、不训练。
"$PYTHON" scripts/train.py stage=stage3 controller=without_root loss=main_replay --cfg job
```

## 结果、恢复与推理

`experiments/实验名_日期/stageN/` 保存 `config.yaml`、配置快照、protocol/source哈希、train/val数据索引、验证JSON、状态及report.txt。大权重和原始rollout在 `${FRANKENMOTION_ARTIFACTS}/实验名_日期/stageN/`；不向Git提交大缓存或视频。

阶段二/三固定每类10个等间距命令点（20类共200条），每类验证文本、种子跨检查点固定，50步纯噪声DDIM、不注入GT根轨迹。保存动作会重算FK/命令量；选择宏平均归一化MAE最佳权重。阶段一按完整验证loss选best。低MAE不表示自然性或接触可靠。

```bash
bash scripts/train_stage2.sh experiment=charlie20 run_date=20261009 "${DATA_ARGS[@]}" runtime.resume=/absolute/path/stage2/latest.pt
```

恢复要求配置、损失、数据索引哈希和源码一致，保存并恢复优化器、采样器和Python/NumPy/Torch/CUDA随机状态。`runtime.stop_after=N` 可作短程恢复检查；它不会假称训练完成。已有运行目录默认拒绝覆盖。恢复到同一report目录保留扫描历史。

新阶段检查点不冒充旧的 `frankenmotion_shared20_v1` 发布包。使用配套入口生成：

```bash
"$PYTHON" scripts/infer.py controller=with_root checkpoint=/path/stage3/best.pt text_cache=/path/example.npz skeleton=/path/skeleton.npz task=reach command=0.4 output=/path/reference.npz
```

同一个底座文件/配置需与训练一致，文本NPZ只读取local/local_mask/tx，不读取真实motion或quantity。旧发布权重仍使用原 `python -m shared_motion.adapter.infer`。

测试：`python -m unittest discover -s tests -v`。测试包括全部控制器/损失组合、20类指标、缺失任务/数据泄漏拒绝、初始化与冻结、无GT生成、两种架构阶段二/三逐位恢复，以及AST命名规则。测试用合成夹具不属于训练数据或实验效果证据。
