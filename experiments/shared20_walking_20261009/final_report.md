# FrankenMotion 20任务训练交付

训练结束：2026-10-09（Asia/Shanghai）。阶段二70000步、阶段三8800步均完成，未延长；最佳命令跟随权重为阶段三7700步。**命令控制指标改善，但自然性未通过本次骨架/轨迹检查。**

## 权重与文件

- [本机最佳检查点](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/checkpoint_007700.pt)（含root/task adapter、优化器与协议；官方backbone另行加载）
- Betail最佳检查点：`/mnt/sda2/frankenmotion/outputs_amass/shared20_walking_20261009_perf/stage3/checkpoint_007700.pt`
- SHA256：`ff5ec6f3f2a7113d158bd7da31a30d2035f098b7e9de3746ef83e3b2e86c81b8`
- 官方backbone SHA256：`c9dca1988dd08dd9e2164ac4cf6ae8fece23011a6371e83d502cdfabe5352e18`
- [全部视频索引](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/index.html)；[turn左](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/turn_left.mp4)、[turn右](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/turn_right.mp4)、[sidestep](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/sidestep.mp4)、[strike](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/strike.mp4)、[kick](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_final_review_20261009/kick.mp4)
- [阶段二完整曲线](/Users/andy/.codex/worktrees/7a7d/frankenmotion/experiments/shared20_walking_20261009/stage2/figures/normalized_mae.png)、[阶段三完整曲线](/Users/andy/.codex/worktrees/7a7d/frankenmotion/experiments/shared20_walking_20261009/stage3/figures/normalized_mae.png)、[阶段三20类响应](/Users/andy/.codex/worktrees/7a7d/frankenmotion/experiments/shared20_walking_20261009/stage3/figures/response_curves.png)
- 视频原件在Betail结果根`review_best_007700/`，未做重定位、稳定、插值、落地或姿态修正。20fps固定相机骨架和足/根轨迹，红左蓝右。

## 训练与审计

- 代码43bf05f；30000步由5bc1060迁移，父检查点和旧protocol保留。数值/梯度对照与真实检查点连续/分段恢复验证通过。实际每万步约5.34分钟（原32.5分钟）。
- 20类共享TaskControl；官方backbone与独立RootControl冻结；只有walk/back_walk/turn接收root请求。架构、batch与loss按既定协议。
- 阶段二8次/1600条、阶段三9次/1800条rollout均已FK重算；优化器步数、冻结、源码及权重血缘核验通过。
- 阶段二全部12030条训练记录均被采样。阶段三固定8800步、按类均衡有放回抽样，实际使用11642/12030条（96.77%）；未追加步数补覆盖。
- 阶段二50000步最优宏MAE0.4705629；60000/70000分别0.4851771/0.5253041，因此以50000进入阶段三。阶段三初始化参数和200条初始rollout与选定存档逐位一致。
- 阶段三7700步宏MAE0.0777231，较阶段二最佳下降83.48%；8800步为0.1017061。选定7700是依据固定扫描最小宏MAE，未按自然性重新挑选。
- 7700步完整重建验证loss0.445449（初始0.124725）；与自由生成训练目标不同，不能用低命令MAE掩盖此变化。

## 自然性与局限

- turn角MAE0.40989rad，步速MAE0.22215m/s；小角请求仍可过转。sidestep位移MAE0.02260m，但抽帧与足迹呈滑移迹象。
- walk/back_walk/jog在这组扫描中，双踝水平速度同时超过0.1m/s的比例中位数为100%；turn为94.6%，sidestep为92.4%。这是描述性证据，不是有真实接触标签的滑步率。
- march根净位移中位数1.430m；twist0.561m、stretch0.421m，也存在明显平动；是否符合各条原始caption需单独判断。没有擅自扩大root任务或修改loss。
- squat、bow、clap的主要姿态变化在抽帧中可辨；point高命令仍明显不足。周期、时序以及身体部位语义不能仅由最大幅度指标确认。
- 已为63条低/中/高命令样本生成21段骨架视频；对所有任务检查起/中/末抽帧和轨迹，对全部200条统计。没有宣称逐帧人工观看所有视频或通过蒙皮/物理自然性评测。
- 新增9类主要依caption及时间标注筛选，尚未逐条人工审核；march验证仅一个独立标注来源；本次固定验证每类10个点，不能等同广泛文本/种子泛化。

## 最佳权重逐类误差

| 任务 | 原始量纲MAE | 归一化MAE |
|---|---:|---:|
| raise_hand | 0.022914 | 0.057286 |
| reach | 0.008499 | 0.028329 |
| strike | 0.014041 | 0.014041 |
| wave | 0.008997 | 0.064266 |
| turn | 0.409893 | 0.058556 |
| sidestep | 0.022598 | 0.028248 |
| back_walk | 0.026298 | 0.047815 |
| kick | 0.028657 | 0.063682 |
| jump | 0.018263 | 0.060876 |
| lean | 0.023384 | 0.046769 |
| walk | 0.144441 | 0.240735 |
| squat | 0.010670 | 0.030486 |
| bow | 0.040331 | 0.057616 |
| clap | 0.042657 | 0.085315 |
| point | 0.143032 | 0.357579 |
| stretch | 0.014018 | 0.028037 |
| twist | 0.021793 | 0.043586 |
| march | 0.013317 | 0.083231 |
| jog | 0.077553 | 0.064628 |
| arm_circle | 0.065368 | 0.093383 |
