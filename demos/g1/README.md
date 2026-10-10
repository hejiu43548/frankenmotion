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

Demo 脚本统一使用 `demos/g1/scripts/`，根目录 `scripts/` 不再保留 demo 文件或兼容链接。`config/` 和 `experiments/g1_demo_N_20261010` 的现有相对链接保持可用。历史报告中的本地 `scripts/g1_demo*.py`、`scripts/reproduce_g1_demo*.sh` 命令应改用 `demos/g1/scripts/` 前缀；远端冻结目录内的 `code/scripts/` 路径保持原样。此次清理不改变动作、控制器、参数和冻结源码。历史交付报告中的“等待验收”是当时状态，最终结论以独立验收记录为准。

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

## Demo3 步态相位优化版

[优化报告](experiments/g1_demo_3_phase_20261010/report.txt)保留与原版的数值和视觉对照。根据支撑脚、摆脚方向和速度选择裁剪点，使用 0.2 秒速度桥；两处 walk→turn 的叉腿停顿被消除，实际最终回归误差约 **3.7 cm**、朝向误差 **2.04°**。仍有 turn→walk 支撑脚位移约 4.6 cm 和转弯低支撑等局限。

```bash
bash demos/g1/scripts/reproduce_g1_demo3_phase.sh physics phase_review_unique
bash demos/g1/scripts/reproduce_g1_demo3_phase.sh full phase_full_review_unique
```

新配置为 `config/g1_demo3_phase.yaml`，冻结版本为 Betail 的 `g1_demo_3_20261010/attempts/phase_matched_final/code`。原始 Demo3 配置、报告和独立验收记录保持原样；优化版已完成自身完整复跑，未冒用原版验收结论。

## Demo3 原生动作的大半径版本

根据用户要求，禁止通过重复步态、逐帧朝向校准或根轨迹缩放扩大半径。新版本从同一个生成器筛选更长的原生圆弧动作，保留原速连续片段，仅做刚体对齐和段间衔接。[报告及真实性检查](experiments/g1_demo_3_native_radius_20261010/report.txt)给出逐帧原生一致性、失败候选与限制。

实际拟合转弯半径约 **1.0 m**（此前约0.37m）；最终回归约 **12.2 cm / 6.28°**。原始任务阈值通过，但末端收稳漂移、turn→walk脚位移和非精确圆轨迹仍需留意。

```bash
bash demos/g1/scripts/reproduce_g1_demo3_native.sh physics native_review_unique
bash demos/g1/scripts/reproduce_g1_demo3_native.sh full native_full_review_unique
```

## 已知局限

这些是经过选种子和场景校准的单场景演示。采用已披露的原生 SONIC 1751/994 控制器替代缺失的发布版 2350 tracker，不能视为泛化或实体机器人验证。

原始 Demo3 实际回归误差约 13.9 cm、朝向误差 2.27°；切换保证支撑和状态连续，但固定时刻裁剪与插值可能在两脚分开时停顿，其步态相位改进见上方独立优化版。其他跟踪误差、低支撑、滑移和软接触穿透见各报告及验收记录。原版结果继续保留可复现记录，优化版差异见独立报告。

## Demo2 返回段相位优化

新增 [相位优化报告](experiments/g1_demo_2_phase_20261010/report.txt) 与独立配置 `config/g1_demo2_phase.yaml`。转弯→回程桥由1.2秒改为0.2秒同支撑脚相位衔接，取消回程静止前缀并减小绕转；实际返回18.3厘米、反向朝向误差0.34°。完整生成/GMR/物理复跑一致。原版配置和冻结入口仍可用。

```bash
bash demos/g1/scripts/reproduce_g1_demo2_phase.sh physics phase_review_unique
bash demos/g1/scripts/reproduce_g1_demo2_phase.sh full phase_full_review_unique
```

Betail新版冻结目录：`/mnt/sda2/frankenmotion/outputs_amass/g1_demo_2_20261010/attempts/phase_return_final`。本版转弯约214°后在回程中回调到180°，保留同样五段动作；14毫秒低支撑和接触滑移峰值等权衡见报告。
