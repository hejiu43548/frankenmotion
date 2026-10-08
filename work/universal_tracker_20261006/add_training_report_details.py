from pathlib import Path
p=Path(__file__).with_name('build_final_report_zh.py');s=p.read_text()
needle="replication=data.get('replication',{})"
block='''page();heading('9  扩展联合训练：配置与开发曲线')
figure(D/'figures/expanded_training/expanded_training.png')
p('曲线只使用开发集，最后点是主候选；自然动作列为严格跟踪，命令列为事件与数值联合通过。不同 checkpoint 均经过相同桌面与抖动资格检查。独立自然测试没有复现开发分数的提升，因此不把这条曲线当泛化证明。','small')
table([['配置','本次实际设置'],['采样与预算','260 环境 ×24 步 ×6,000 次更新，约 3,744 万环境控制步；39 个桌面槽位、65 个扩展槽位、156 个旧动作槽位。'],['优化器','Adam；固定学习率 2e-5；PPO clip 0.2；5 epochs /4 minibatches；γ=0.99，λ=0.95，entropy=0.001。'],['参考观察','当前状态＋未来 0.10/0.20/0.40/0.70/1.00 s；共享输入归一化，无任务类别输入。'],['重置与保留','35% 片段起点，其余均匀参考相位；保留回放 53,644 条训练观测，每次优化采样 256 条，MSE 权重 0.5。'],['共同奖励','全局根位置 / 朝向、身体与关节跟踪、骨盆相对末端位置、速度和双足高度。全部动作使用相同公式与系数。']],[90,405])
p('平滑项作用于物理关节目标变化与参考关节速度之间的残差，同时保留少量标准动作变化惩罚。它是训练奖励，不是部署输出上的逐类滤波。当前最终版本从 v4_3000 继续训练，不能把上述预算当作从零训练的总成本。','small')
'''
assert needle in s;s=s.replace(needle,block+needle);s=s.replace("'躯干高频 RMS','足部切向速度代理'","'躯干高频 RMS (rad/s)','足部切向速度代理'")
p.write_text(s)
