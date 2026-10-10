# G1 连续物理仿真 demos

三个 demo 的实现、Hydra 配置、实验报告和验收记录统一保存在本目录。它们使用同一个完整 20 任务 Stage2 checkpoint，经人体动作生成、GMR 重定向、参考组合和连续 SONIC 力矩控制执行。

| Demo | 动作 | 报告 | 独立验收 |
| --- | --- | --- | --- |
| 1 | 走近 → 踢击 → 手击同一沙袋 | [报告](experiments/g1_demo_1_20261010/report.txt) | [记录](reviews/demo_1_review.json) |
| 2 | 走 → 招手 → 鞠躬 → 转身 → 走回 | [报告](experiments/g1_demo_2_20261010/report.txt) | [记录](reviews/demo_2_review.json) |
| 3 | 走 → 左转 → 走 → 左转 | [报告](experiments/g1_demo_3_20261010/report.txt) | [记录](reviews/demo_3_review.json) |

## 目录

- `scripts/`：三个 demo 的入口、共享生成/物理执行、审计、渲染和复现脚本。
- `config/`：`g1_demo.yaml`、`g1_demo2.yaml`、`g1_demo3.yaml`。
- `experiments/`：原始交付报告、指标、参数快照、关键帧、哈希及复现结果。
- `reviews/`：父对话独立验收记录的归档副本。

原仓库 `scripts/`、`config/` 和 `experiments/g1_demo_N_20261010` 路径保留相对符号链接，兼容已有命令和报告引用。此次整理不改变动作、控制器、参数和冻结源码。历史交付报告中的“等待验收”是当时状态，最终结论以独立验收记录为准。

## 复现

在仓库根目录执行，最后一个参数必须使用未占用的名字：

```bash
bash demos/g1/scripts/reproduce_g1_demo.sh physics demo1_review_unique
bash demos/g1/scripts/reproduce_g1_demo2.sh physics demo2_review_unique
bash demos/g1/scripts/reproduce_g1_demo3.sh physics demo3_review_unique
```

将 `physics` 改为 `full` 会在新目录重新生成、重定向并执行。脚本通过 SSH 使用 Betail 的冻结入口及现存依赖；它们不是自动安装环境的独立发行包。

大文件保留在 Betail：

```text
/mnt/sda2/frankenmotion/outputs_amass/g1_demo_1_20261010/attempts/demo1_final
/mnt/sda2/frankenmotion/outputs_amass/g1_demo_2_20261010/attempts/return_heading_final
/mnt/sda2/frankenmotion/outputs_amass/g1_demo_3_20261010/attempts/symmetric_first
```

各目录的 `continuous.mp4` 为完整实际物理状态录像；`code/` 为冻结入口。人体/GMR 数据位于各 demo 根目录的 `generated/` 和 `retarget/`，不纳入 Git。视频本机位置见各报告。

## 已知局限

这些是经过选种子和场景校准的单场景演示。采用已披露的原生 SONIC 1751/994 控制器替代缺失的发布版 2350 tracker，不能视为泛化或实体机器人验证。

Demo3 实际回归误差约 13.9 cm、朝向误差 2.27°；切换保证支撑和状态连续，但固定时刻裁剪与插值可能在两脚分开时停顿，尚未实现自然的步态相位衔接。其他跟踪误差、低支撑、滑移和软接触穿透见各报告及验收记录。此次目录整理未修改这些行为。
