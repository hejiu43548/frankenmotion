# -*- coding: utf-8 -*-
"""Human-readable local delivery index from final evidence only."""
import json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';data=json.loads((D/'report_data.json').read_text());f=data['frozen'];a=data['controllers']['candidate']['aggregate'];b=data['controllers']['broad']['aggregate'];n=data['controllers']['candidate']['natural']['results'];table=data['controllers']['candidate']['table'];remote='/home/pku/frankenmotion';rd=remote+'/outputs_amass/universal_tracker_20261006';rw=remote+'/work/universal_tracker_20261006';py=remote+'/work/mjlab_stable_env/bin/python';lines=['统一 Tracker 扩展实验交付说明',
'实验窗口：北京时间 2026-10-06 02:19:43–10:19:43',
'',
f'最终共享权重：{f["selected"]}',
f'checkpoint SHA256：{f["checkpoint_sha256"]}',
f'actor SHA256：{f["actor_sha256"]}',
'部署只使用这个共享 actor；没有按动作切换 tracker。生成端仍有既有的命令适配器。训练仍包含一个冻结稳定 actor 的保留正则。',
'demo 为离线生成参考后再执行物理控制，未验证在线端到端生成时延。50 Hz 是仿真控制频率；视频为1×回放。',
'',
f'新噪声 880 条：本轮完成 {a["complete"]}，事件 {a["event"]}，命令联合通过 {a["joint"]}；旧共享版联合通过 {b["joint"]}。',
f'11 类达标：{data["controllers"]["candidate"]["passed_classes"]}/11。',
f'独立来源自然动作：完整执行 {sum(v["complete"] for v in n.values())}/{sum(v["requests"] for v in n.values())}；严格跟踪 {sum(v["accurate_complete"] for v in n.values())}/{sum(v["requests"] for v in n.values())}。',
f'新桌面流程：{table["success"]}/{table["planned"]} 全流程通过。',
'不能将展示视频等同于所有类别或布局成功；当前仍未证明通用动作和真机部署能力。',
'',
'本地文件',
'- frozen_unified/：最终 checkpoint、TorchScript actor 和冻结协议。',
'- release/：源代码快照、环境、推理契约、指标定义、训练与备份清单。',
'- figures/：11 类折线图、分组图、结构图和诊断图，含可编辑 SVG / PDF。',
'- visuals/：实际物理轨迹视频与视觉检查图；视频索引为 video_index.json。',
'- tables/：逐条880测试、分类汇总与扰动结果 CSV。',
'- report_data.json：冻结测试、对照、诊断及附加 seed 的汇总。',
f'- 中文报告：{ROOT}/output/pdf/universal_tracker_20261006/统一Tracker扩展实验报告_中文.pdf',
'',
'压缩包内中文报告：报告/统一Tracker扩展实验报告_中文.pdf',
'远程完整证据',rd,'原始轨迹、全部失败、所有 checkpoint、训练日志和生成样本均保留。',
f'原 tracker 备份：{rd}/backup/stable_frozen 与 prior_unified_frozen。',
'没有 push。',
'',
'最小复现：重新执行已冻结的 880 条参考（使用新的输出名，避免覆盖证据）',
f'cd {remote}',shlex.join([py,rw+'/evaluate_cpu_general.py',
'--checkpoint',rd+'/frozen_unified/policy.pt',
'--manifest',rd+'/fresh_final/native_manifest.json',
'--name',
'reproduce_frozen_880',
'--workers',
'4']),shlex.join([py,rw+'/assess_native_eleven.py',
'--name',
'reproduce_frozen_880']),'重新执行时另换未使用的 --name；原数据目录已保留，不需要删除。环境与资产依赖以 release/inventory.json、python_environment.txt、inference_contract.json 为准。',
'',
'完整训练复现',
'选中候选的训练参数、初始 checkpoint、源码哈希与采样记录见 release/inventory.json 中 joint_v5_expanded 条目。附加 seed 6107 只复现扩展阶段，共享 v4 起点；不是全流程训练方差。',
'完整 pipeline 需要远程原仓库的基础模型、GMR/机器人资源和生成命令适配器；这个源代码包不是脱离原仓库即可运行的独立软件发行版。']
if data.get('replication',{}).get('results'):
 r=data['replication']['results'];lines += ['',
f'附加seed6107：880条完成{r["aggregate"]["complete"]}，联合通过{r["aggregate"]["joint"]}；桌面{r["table"]["success"]}/{r["table"]["planned"]}。replication_seed6107为附加研究权重，不替换frozen_unified主权重。']
index=D/'video_index.json'
if index.exists():
 lines+=['',
'正式展示视频']
 for r in json.loads(index.read_text())['videos']:lines += [r['title'],r['local_path'],r.get('scope',
'')]
if index.exists():
 lines+=['','探索性与诊断视频（含明确失败项）']
 for r in json.loads(index.read_text()).get('diagnostic_videos',[]):lines += [r['title'],str(Path(r['local_path']).relative_to(D)),r.get('scope','')]
(D/'交付说明.txt').write_text('\n'.join(lines)+'\n');print(D/'交付说明.txt')
