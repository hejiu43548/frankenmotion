# Kick 数据修复候选与参数监督覆盖（2026-10-09）

当前候选33train/2val，28/2原始录制；Betail来源为`outputs_amass/source_repair_kick_20261009_v1`。BABEL帧级kick和明确单动作序列经过实际右踝前向峰值定位，无补帧、镜像、重定时或幅度缩放；保留其他19类记录。HDM05优先，但不得将BABEL时间边界称为已核验官方cuts。

35缓存逐条精确裁剪与控制量重算核验通过，10条幅度分层训练视频20fps/帧数核验；6项回归测试通过。逐条语义尚待用户复核。视频为FK骨架，抽帧检查不等于穷尽动作认证。

## 监督缺口

目标0.25–0.70m；仅10/33训练落在范围内，5/10等宽区间空缺。2个验证均在范围外。真实训练量0.434–1.044m，并不能证明已学会小幅度前踢。完整数量和独立录制分布见`coverage/coverage.json`及`coverage/report.md`。原始strike同样只有3/13训练处于1.5–2.5m/s，7/10区间空缺，范围内验证0条。

用户决定：strike带混合拳残留和稀疏监督缺陷暂定试用；先清洗kick，之后统一训练。当前没有训练，不自动修改正式控制范围。

脚本与配置：`data_processing/kick/`、`data_processing/event_text.py`、`data_processing/parameter_coverage.py`、`config/kick_repair.yaml`、`config/kick_review.yaml`、`config/parameter_coverage.yaml`。大型视频在本机`outputs_amass/task_data_review_20261009/kick/`，复核页http://127.0.0.1:50086/kick/index.html。
