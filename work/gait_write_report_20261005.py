import json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';out=D/'report';out.mkdir(exist_ok=True);f=json.loads((D/'final_results.json').read_text());g=f['selected'];c=f['selected_cpu'];comp=json.loads((D/'development_comparison.json').read_text());p=json.loads((D/'frozen/protocol.json').read_text());parity=json.loads((D/'frozen/parity_audit.json').read_text());visual=json.loads((D/'visual_review/review.json').read_text())
s=f'''FrankenMotion → G1 正常步态与桌面交互改进实验

结果
冻结后的同一个 tracker 在32个全新随机桌位上，GPU物理仿真成功{g['successes']}/32，独立CPU物理仿真成功{c['successes']}/32。GPU平均到达误差{100*g['mean_root_error_m']:.2f}cm，平均手部目标误差{100*g['mean_palm_error_m']:.2f}cm，最短连续桌面接触{g['min_contact_s']:.2f}s；CPU对应为{100*c['mean_root_error_m']:.2f}cm、{100*c['mean_palm_error_m']:.2f}cm、{c['min_contact_s']:.2f}s。完整保留32个随机布局及两套执行结果，没有删除失败例或重新抽样。

这次解决了什么
上一版到达/接触成功率高，但腿部运动被tracker压缩成贴地小碎步；固定6秒走一小段距离也会加重这个问题。本轮保留原有方向/距离条件生成器，主要改动执行层：参考行走时长改为clip(距离/.45+.8,2.8,4.8)秒，并训练一个共享tracker学习交替抬脚和稳定交互。参考空间路径没有被修改；视频没有快放。

六个相同开发场景，三组对比（都是GPU物理执行）
                          上一版       仅调整参考时长       新版单一tracker
膝关节摆动/参考幅度        41.3%           51.0%              98.6%
脚底离地95分位            0.52cm          1.09cm             2.66cm
双脚同时支撑比例          77.4%           59.8%              34.1%
接触点平均滑移速度        4.39cm/s        11.52cm/s          4.70cm/s
平均到达误差              0.58cm          1.35cm             2.70cm
平均手部误差              1.87cm          1.54cm             2.63cm
交互成功                  6/6            6/6                6/6
以上是工程诊断指标，不是经验证的人类感知自然度评分。滑移相比原版没有明显降低，但避免了只加速带来的大幅滑移；到达精度略下降。脚底高度计算覆盖全部脚部碰撞几何，非脚踝中心高度。

训练和尝试
1. 增加脚部位置/速度/姿态、膝关节、脚底离地和支撑脚速度奖励，做了两轮PPO。部分检查点抬脚改善，但有保持桌面接触变差的权衡，未采用为最终权重。
2. 测试接触约束重定向、单独SONIC、SONIC配手部反馈等。最终未采用这些额外方案。
3. 采用SONIC行走教师和上一版交互教师收集16条训练参考上的物理数据，在停步段平滑衔接教师。蒸馏到同一个361输入的MLP。第一版在GPU通过，但独立CPU发生过跌倒，没有交付。
4. 在8条训练参考上收集学生参与执行的DAgger数据（70%学生、30%教师，仅用于训练），再蒸馏得到最终第二版。教师验证组14/15没有进入DAgger采集。
最终部署只加载一个actor，推理没有教师、动作混合、任务路由或按阶段换权重。训练期使用多个教师不等于部署使用多个策略。64条最终轨迹的已记录动作与同一个导出actor一致，最大差异{parity['max_error']:.3g}。

演示完整pipeline
已知仿真桌子坐标 → 计算停止位置的方向/距离 → 现有FrankenMotion条件分支生成行走 → 最多3次相同噪声的输入命令修正 → uniform GMR → 行走参考时间重采样 → 显式平滑停步和右手抬起/前伸/放下/保持IK → 单一tracker在线控制G1 → MuJoCo物理。
放手包含显式场景IK，不是声称整个交互由扩散模型端到端生成。新的goal adapter本轮没有再修改。桌位是从仿真坐标获得，不包含视觉感知。

测试范围
冻结权重后使用种子75005000生成32个随机布局，停止距离0.85–1.75m、方向±0.5rad，桌子中心再向前0.65m，桌高0.8m。固定walk/reach文字模板，名义物理参数。成功标准：完整执行、到达误差<20cm、手目标误差<10cm、连续至少1秒的50Hz采样桌面顶部接触（法向力>0.2N）。从保存的qpos/qvel/ctrl和原始模型独立复算接触。
实际机器人状态只有初始化写入，之后由actor输出和物理积分产生。渲染时重放保存状态是画面生成，不是控制时写参考状态。

视觉验收
{visual['summary']}
检查范围与具体文件见visual_review/review.json。视频为1倍速度；前后对比中较短的行走结束后定格并标注，未对视频做时间压缩。

仍有局限
步态已恢复清楚的左右交替抬脚，相比旧版拖步有明显改善，但手臂摆动仍偏小、停止前有轻微后仰，不能称为完全人类化步态。名义仿真成功不代表真机稳定，未进行硬件测试。本轮没有重跑此前11类动作的完整能力保留基准，因此不能把桌面demo的结果外推为所有技能都提升。

文件与复现
videos/funding_demo.mp4：完整随机场景和前四个场景。
videos/gait_before_after.mp4：同一开发场景的原版/新版真实速度步态对比。
report/gait_comparison.png：三组定量比较。
report/final_results.png：全部32个GPU/CPU结果。
weights/policy.pt SHA256：{p['weights']['policy.pt']}
weights/actor.pt SHA256：{p['weights']['actor.pt']}
完整源代码、权重、训练样本、原始状态/控制和复现命令见发布包README.txt。原有生成器资产和已配置环境保留在服务器，发布包不是可自动迁移的完整训练环境。未push，未覆盖上一版发布包。
'''
(out/'实验说明.txt').write_text(s);print(out/'实验说明.txt')
