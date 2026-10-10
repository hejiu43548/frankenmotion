# 阶段二最终70000步：20任务评估

主评估权重：`/mnt/sda2/frankenmotion/outputs_amass/shared20_walking_20261009_perf/stage2/checkpoint_070000.pt`
SHA256：`52a787652b9ee20f56678a6f5949a835fbfe10232f3a64dbabffcde9f083002c`

结论：阶段二已经存在明显的控制不足与滑动，最终70000步并不优于按命令MAE挑选的50000步。不能把问题全部归因于阶段三。

- 宏归一化MAE：50000=0.470563，70000=0.525304（恶化11.6%）。
- 20类宏候选支撑滑动占比：50000=45.7%，70000=54.8%。
- 13/20类的端点响应增益低于0.3（理想值约1，负值表示反向），许多任务只会生成相近幅度的动作。
- turn的原生步速MAE：50000=0.1113m/s，70000=0.1160m/s。

[全部20类视频索引](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/index.html)。每类低/中/高3点，turn左右各3点，共63条动作、21段视频。上方是固定相机原始FK骨架，下方是脚/根世界坐标轨迹；20fps，无稳定、重定位或落地修正。

下表是固定验证文本/种子10命令点的诊断，不等于多文本泛化。滑动占比采用此前同一保守接触代理（低脚、低竖直速度、持续0.15s，允许踝/足部滚动），没有接触真值，不能解读为精确接触滑步率。端点响应增益不测时序/语义完整性。

|任务|MAE（原量纲）|归一化MAE|端点响应增益|估计滑动占比|判断与视频|
|---|---:|---:|---:|---:|---|
|raise_hand|0.154 m|0.385|0.21|32.2%|幅度变化弱，整体偏高；已有滑动。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/raise_hand.mp4)|
|reach|0.218 m|0.727|-0.01|16.1%|几乎不响应命令，幅度不足。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/reach.mp4)|
|strike|1.315 m/s|1.315|-0.02|94.6%|速度控制基本失效，且支撑脚滑动严重。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/strike.mp4)|
|wave|0.063 m|0.449|0.08|29.1%|命令变化对幅度影响很小。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/wave.mp4)|
|turn|0.541 rad|0.077|0.78|74.2%|有转角响应，小角请求过转；支撑脚滑动明显。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/turn_right.mp4)|
|sidestep|0.468 m|0.585|-0.16|22.2%|响应方向反了：请求越大，实测侧移反而减小。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/sidestep.mp4)|
|back_walk|0.374 m/s|0.680|0.13|62.6%|速度变化弱，滑动明显。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/back_walk.mp4)|
|kick|0.125 m|0.277|0.08|76.7%|踢腿幅度几乎固定，未学到有效可调控制。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/kick.mp4)|
|jump|0.382 m|1.272|0.09|79.2%|高度严重不足，响应弱。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/jump.mp4)|
|lean|0.437 rad|0.873|0.45|72.9%|响应幅度压缩且整体不足，滑动明显。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/lean.mp4)|
|walk|0.125 m/s|0.209|0.28|68.9%|步速范围压缩，滑动明显。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/walk.mp4)|
|squat|0.240 m|0.685|0.42|58.9%|整体下蹲幅度偏大，控制范围压缩。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/squat.mp4)|
|bow|0.056 rad|0.080|1.06|24.1%|命令跟随相对较好，仍有根/足部漂移。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/bow.mp4)|
|clap|0.136 m|0.272|0.13|21.0%|手部幅度近乎固定，控制弱。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/clap.mp4)|
|point|0.236 m|0.591|0.79|9.6%|有响应但整体偏差大；足部问题相对较轻。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/point.mp4)|
|stretch|0.091 m|0.182|1.06|49.5%|有较明确幅度响应，仍有足部漂移。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/stretch.mp4)|
|twist|0.135 rad|0.271|0.16|69.4%|转动幅度控制弱，滑动明显。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/twist.mp4)|
|march|0.048 m|0.303|0.89|74.6%|抬脚幅度有响应，但滑动严重且验证仅单来源。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/march.mp4)|
|jog|1.191 m/s|0.992|0.09|100.0%|实际速度远低于请求，脚步运动与位移不匹配。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/jog.mp4)|
|arm_circle|0.197 m|0.281|0.12|60.7%|垂直幅度变化弱；不能由此证明完整画圈语义。 [视频](/Users/andy/Develop/frankenmotion/outputs_amass/shared20_stage2_review_20261009/arm_circle.mp4)|

## 直接可见的例子

- sidestep请求0.400/0.756/1.200m，生成0.387/0.350/0.258m：幅度控制反向，视频中仍有明显上肢动作，不能因看起来在运动就当作侧步完成。
- turn右转请求0.389/1.944/3.500rad，生成1.688/2.458/3.147rad：小角明显过转，端点角度接近也不能代表自然步态。
- bow与stretch的幅度响应相对明显，但此处没有把它们标记为自然性通过。

## 文件

- 统计来源stage2_assessment.json；与50000对照沿用上次sliding_audit_20261009的同一算法。
- 已检查turn/sidestep的起中末骨架帧及轨迹，其他任务上述判断基于10点量化指标；没有冒称逐帧人工看完所有视频。
- Betail视频原件：`/mnt/sda2/frankenmotion/outputs_amass/shared20_walking_20261009_perf/review_stage2_070000/`。
- 本次仅评估和渲染，无新训练、无参数或数据修改，原监控继续PAUSED。
