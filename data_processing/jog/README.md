# jog 当前数据审计（2026-10-10）

本轮仅检查，**未清理或替换训练数据、未训练**。当前输入为已通过jump V3清理的完整任务数据：`source_repair_jump_20261010_v3`。其中jog与已完成的shared20_clean5训练数据逐条相同。用户此前通过jump，strike/kick暂定通过的决定保持不变。

当前定义见 `shared_motion/adapter/catalog.py`：向前、稳定放松的慢跑，控制量为根水平路径速度，范围0.8–2.0m/s；它不是原地跑。弧线运动不因净位移小而直接判错，审计采用路径长度速度。

## 库存和全量结果

| 项目 | train | val | 合计 |
| --- | ---: | ---: | ---: |
| 当前jog片段 | 742 | 92 | 834 |
| 独立录制 | 363 | 44 | 407 |
| 其中HDM05片段 | 50 | 12 | 62 |
| 速度低于0.2m/s | 437 | 49 | 486 |
| 速度低于命令下限0.8m/s | 579 | 70 | 649 |
| 在0.8–2.0m/s内 | 148 | 21 | 169 |
| 超过2.0m/s | 15 | 1 | 16 |
| 有run/jog帧标签但当前裁剪完全错过 | 30 | 4 | 34 |
| 超过半段被标为walk且没有run/jog | 17 | 5 | 22 |
| 超过四分之一被标为其他冲突动作且没有run/jog | 14 | 0 | 14 |
| 未找到该来源的run/jog帧标签 | 565 | 61 | 626 |

这些诊断集合有交集，不能相加当错误率。缺少BABEL标注不等于动作错误；低速也可能是真正原地跑，只是不符合当前任务的前进速度语义。486条低速记录主要来自BioMotionLab（368）、BMLmovi（72），另外CMU20、HDM05 20、DFaust4、MPI_Limits2。

834缓存哈希检查通过，动作与完整来源切片逐元素相同，根路径速度重新FK计算与缓存标签最大误差4.77e-7m/s。训练速度中位数0.169m/s、验证0.184m/s。问题主要在数据筛选与语义，而不是这批缓存的速度计算错误。

## 已渲染核对的具体问题

[6条诊断视频](http://127.0.0.1:50086/jog_audit/index.html)，全部来自现有TRAIN缓存，只做FK渲染，没有稳定、修姿或重新裁剪；按问题类型挑选，不用于估计总体错误率。

1. `CMU/31/31_09_poses`：0–4.05s，0.026m/s。关键帧为站立摆手，并非跑步；整段描述出现“pretending to run”，正则回退将其纳入。
2. `MPI_HDM05/dg/HDM_dg_01-04_02_120_poses`：20.15–23.95s，0.774m/s。BABEL该段约97%为walk，跑步覆盖为0；旧动作文字把“jogging, stopping, walking slowly back”混在一个区间里。实际关键帧显示走路。
3. `Eyes_Japan_Dataset/aita/walk-07-moonwalk-aita_poses`：4.45–9.45s，0.731m/s。BABEL为dance/perform，腿部文字写“running backward and sliding”，正则使月球步/舞蹈进入jog。
4. `MPI_HDM05/dg/HDM_dg_01-03_02_120_poses`：27.80–31.60s，0.036m/s。实际结束姿势为T-pose；BABEL是transition/T-pose，旧描述仍将该时间段写作jogging。
5. `ACCAD/Male2Running_c3d/C7 - run backwards t2_poses`：0–3.20s，2.831m/s。确实是倒跑，但不符合当前向前慢跑定义。
6. 对照 `MPI_HDM05/tr/HDM_tr_01-03_03_120_poses`：20–25.35s，1.558m/s，BABEL跑步帧覆盖100%；关键帧有连续跑步动作。此对照不等于已完成准入验收。

各视频检查5个等距关键帧，并校验视频20fps及帧数；没有宣称全量834条逐帧人工观看。

## 产生污染的路径

`scripts/prepare_amass20.py` 先用整段caption的 `\bjog|\brun(?:ning)?\b` 找来源，再用相同正则匹配身体部位/动作描述的区间。多个部位区间合并后切成2–6秒；短段扩到至少2秒，缺少局部匹配则退回整段标注范围。

因此“假装跑”“腿部后滑”“慢跑后走回来”可以通过，即使区间里实际是别的动作。52条使用了整段caption回退；22条虽有局部匹配却没有action部位的匹配。旧脚本没有BABEL类别门槛，没有持续前进、步态、停顿或冲突动作准入；根路径速度本身也不区分走路、舞蹈、倒跑和慢跑。

## 更可靠标注库存与限制

本地BABEL train/val/extra中，按当前项目family split且原始AMASS文件存在，可找到367/42条run/jog帧事件，涉及181/24个录制。其中HDM05为88/19条事件、38/7个录制。这是可调查的事件库存，不是清理后数量；可能仍有原地、倒跑和多个标注者的重复事件。未核实官方jogLeftCircle/jogRightCircle cuts映射。

还发现两个来源中的23条标签把多个不同动作都写成0–1s，而录制超过5s。这些时间信息本轮单列为不确定，不参与精确覆盖统计，原记录保存在 `uncertain_babel_timing.json`。所以最终“裁剪错过跑步标签”为34条，替代初次直接信任全部时间字段时的33条。不能只因带act_cat就假设时间边界可靠。

后续修复应从可信run/jog类别事件出发，核对标注时间单位，裁取连续跑步段，排除站立/走路/舞蹈/跳跃/倒跑；结合身体朝向速度、左右交替步态及腾空证据验证，明确区分原地跑与移动慢跑。腾空代理不能单独判跑步真伪，慢跑过渡段和标注空缺需单独处理。本轮没有执行这些数据替换。

## 复现

审计脚本 `data_processing/jog/audit.py`、渲染 `review.py`；配置 `config/audit_jog.yaml`。结果在Betail `/mnt/sda2/frankenmotion/outputs_amass/jog_data_audit_20261010`；本地小型记录在 `.codex/jog_audit_20261010/`。

```bash
export PYTHONPATH=/tmp/frankenmotion_hydra_deps:.
PYTHON=/home/psirobot/projects/frankenmotion/.venv_unified/bin/python
"$PYTHON" -m data_processing.jog.audit
"$PYTHON" -m data_processing.jog.review
```

结果包含summary、834条完整逐条审计、BABEL帧候选、时间不确定标签、来源哈希及6条原始动作诊断视频。


后续：用户授权 jog/march 联合修复，最新结果见 [../jog_march/README.md](../jog_march/README.md)。本页保留原始检查结果。
