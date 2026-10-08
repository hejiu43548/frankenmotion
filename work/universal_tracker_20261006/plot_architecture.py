"""Editable vector diagram of the implemented pipeline, including training provenance."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parents[2];out=R/'outputs/universal_tracker_20261006/figures/architecture';out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(13.8,7.4));ax.set_xlim(0,14);ax.set_ylim(0,8);ax.axis('off')
def box(x,y,w,h,title,body,color='#eaf1f7',dashed=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.08',facecolor=color,edgecolor='#33566d',linewidth=1.1,linestyle='--' if dashed else '-'))
 ax.text(x+.14,y+h-.21,title,fontsize=11.2,fontweight='bold',va='top',color='#183b52');ax.text(x+.14,y+h-.65,body,fontsize=9.2,va='top',linespacing=1.55,color='#233a4a')
def arrow(a,b,label=None,dashed=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,linewidth=1.2,color='#426b80',linestyle='--' if dashed else '-'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.11,label,fontsize=8.4,ha='center',color='#426b80')
ax.text(.15,7.8,'Parameterized motion generation to one shared G1 tracker',fontsize=18,fontweight='bold',color='#183b52')
box(.15,4.9,2.55,2.05,'1  FrankenMotion','Text + numeric commands\nFrozen diffusion backbone\nExisting command adapters\n20 Hz human motion')
box(3.1,4.9,3.15,2.05,'2  Robot reference','GMR + fixed reference processing\nClip placement / timing / blends\nRelative-command SE(2) anchoring\n50 Hz G1 reference')
box(6.65,4.9,3.45,2.05,'3  Shared policy','Current state + reference preview\nMLP: 512 / 256 / 128, ELU\n29 normalized joint actions\nNo task ID or policy routing','#e5f3ed')
box(10.5,4.9,3.25,2.05,'4  Physical execution','Joint targets + native PD\nMuJoCo dynamics / contacts\nActual state and motor limits\n29 DoF G1, fixed hands','#fff1de')
for a,b in [((2.78,5.95),(3.02,5.95)),((6.33,5.95),(6.57,5.95)),((10.18,5.95),(10.42,5.95))]:arrow(a,b)
arrow((11.4,4.82),(11.4,4.35));arrow((11.4,4.35),(8.4,4.35),'proprioception');arrow((8.4,4.35),(8.4,4.82))
ax.text(.15,4.35,'Training only',fontsize=13,fontweight='bold',color='#183b52')
box(.15,1.65,3.5,2.1,'Joint reference corpus','Old motions + full tabletop sequences\nExpanded trials: official natural motions\nSource-file train / val / test separation\nTask-balanced sampling; no task input','#f2f3f5',True)
box(4.15,1.65,3.75,2.1,'Joint PPO optimization','Pose / root / velocity tracking rewards\nActuation and physical smoothness costs\nTraining critic and randomized resets\nOne shared set of actor parameters','#f2f3f5',True)
box(8.4,1.65,3.65,2.1,'Retention regularizer','One frozen stable actor\nRecorded state/action replay\nActor MSE added during PPO\nTeacher absent from deployment','#f2f3f5',True)
arrow((3.73,2.65),(4.07,2.65));arrow((8.32,2.65),(7.98,2.65));arrow((6.0,3.83),(6.0,4.5));arrow((6.0,4.5),(7.35,4.5));arrow((7.35,4.5),(7.35,4.82))
ax.text(.15,.98,'Provenance: BeyondMimic-derived control/training; historical SONIC teacher lineage; this round retains a frozen stable actor.',fontsize=9.6,color='#415c6b')
ax.text(.15,.58,'Reference anchoring updates future coordinates only. No simulator pose overwrite, diffusion replanning, hand grasping, or hardware claim.',fontsize=9.6,color='#415c6b')
ax.text(.15,.18,'Natural-motion expansion is an experimental training stage; the final selected checkpoint and its corpus are reported separately.',fontsize=9.6,color='#415c6b')
zh='--zh' in sys.argv
if zh:
 from matplotlib.font_manager import FontProperties
 translations={
 'Parameterized motion generation to one shared G1 tracker':'参数化动作生成 → 单一共享 G1 Tracker',
 '1  FrankenMotion':'1  FrankenMotion 动作生成',
 'Text + numeric commands\nFrozen diffusion backbone\nExisting command adapters\n20 Hz human motion':'文本 + 数值命令\n冻结的扩散模型主干\n已有命令适配器\n20 Hz 人体动作',
 '2  Robot reference':'2  机器人参考动作',
 'GMR + fixed reference processing\nClip placement / timing / blends\nRelative-command SE(2) anchoring\n50 Hz G1 reference':'GMR + 固定参考处理\n片段放置 / 重定时 / 过渡\n相对命令边界坐标校正\n50 Hz G1 参考',
 '3  Shared policy':'3  共享控制策略',
 'Current state + reference preview\nMLP: 512 / 256 / 128, ELU\n29 normalized joint actions\nNo task ID or policy routing':'当前状态 + 未来参考\nMLP：512 / 256 / 128，ELU\n29 维归一化关节动作\n无任务 ID 或策略切换',
 '4  Physical execution':'4  物理执行',
 'Joint targets + native PD\nMuJoCo dynamics / contacts\nActual state and motor limits\n29 DoF G1, fixed hands':'关节目标 + 原生 PD\nMuJoCo 动力学与接触\n真实状态与执行器限幅\n29 DoF G1，固定手部',
 'proprioception':'本体感知反馈','Training only':'仅在训练时使用',
 'Joint reference corpus':'联合参考语料',
 'Old motions + full tabletop sequences\nExpanded trials: official natural motions\nSource-file train / val / test separation\nTask-balanced sampling; no task input':'旧动作 + 完整桌面交互序列\n扩展实验：官方自然动作\n自然动作按源文件隔离数据集\n按组平衡采样；不输入任务类别',
 'Joint PPO optimization':'联合 PPO 优化',
 'Pose / root / velocity tracking rewards\nActuation and physical smoothness costs\nTraining critic and randomized resets\nOne shared set of actor parameters':'姿态 / 根轨迹 / 速度跟踪奖励\n动作与物理平滑性约束\n训练 critic 与随机初始化\n一套共享 actor 参数',
 'Retention regularizer':'旧能力保留正则',
 'One frozen stable actor\nRecorded state/action replay\nActor MSE added during PPO\nTeacher absent from deployment':'一个冻结的稳定 actor\n历史状态 / 动作回放\nPPO 中加入 actor MSE\n部署时不包含教师',
 'Provenance: BeyondMimic-derived control/training; historical SONIC teacher lineage; this round retains a frozen stable actor.':'继承关系：BeyondMimic 控制与训练框架；历史 SONIC 教师来源；本轮保留一个冻结稳定 actor。',
 'Reference anchoring updates future coordinates only. No simulator pose overwrite, diffusion replanning, hand grasping, or hardware claim.':'坐标校正只更新未来参考；没有改写仿真状态，也未实现扩散重规划、灵巧手抓取或真机部署。',
 'Natural-motion expansion is an experimental training stage; the final selected checkpoint and its corpus are reported separately.':'自然动作扩展属于实验训练阶段；最终选中的 checkpoint 及其训练语料在报告中单独说明。'}
 for text in ax.texts:
  text.set_text(translations.get(text.get_text(),text.get_text()));text.set_fontproperties(FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf',size=text.get_fontsize()))
fig.tight_layout(pad=.6)
for ext in ['png','pdf','svg']:fig.savefig(out/(('pipeline_zh' if zh else 'pipeline')+'.'+ext),dpi=190,bbox_inches='tight',facecolor='white')
print(out)
