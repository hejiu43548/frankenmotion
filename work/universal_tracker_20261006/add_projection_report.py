from pathlib import Path
p=Path(__file__).with_name('build_final_report_zh.py');s=p.read_text();needle="page();heading('10  Demo、命令含义与复现入口')"
block='''study=data.get('projection_study')
if study and study.get('projection_audit'):
 page();heading('9  探索性对照：共享策略生成物理训练参考')
 pa=study['projection_audit'];p(f'冻结正式测试后追加的假设探索：同一个主策略执行全部 307 条新增训练来源；完整执行 {pa["physical_complete"]} 条，其中 {pa["accepted"]} 条同时满足平均全局身体偏差 ≤0.20 m、逐帧平均偏差 P95 ≤0.40 m。只用这些实际轨迹替换对应训练参考，其余保留原参考。没有按动作另训教师。')
 p('两支训练使用相同主策略起点、seed 6108、2,000 次更新、优化器、采样比例与奖励。对照支继续使用原始 1,084 条参考；实验支仅替换上述通过筛选的参考，片段数量与时长不变。只评估开发集，不重选正式权重，不再次使用最终测试挑结果。')
 if study.get('results'):
  rr=study['results']['results'];rows=[['开发评估','原始起点','原参考续训','物理参考续训']]
  rows.append(['11类联合通过 /110',*[str(rr[tag]['eleven']['joint']) for tag in ['initial','raw','projected']]])
  rows.append(['自然动作严格通过 /57',*[str(natural_totals(rr[tag])['accurate_complete']) for tag in ['initial','raw','projected']]])
  rows.append(['桌面通过 /6',*[str(rr[tag]['table']['success']) for tag in ['initial','raw','projected']]])
  table(rows,[200,98,98,99]);p('这是主测试结果已知之后的一次开发探索，不是新的确认性证据。原始起点的11类数值来自原生 CPU 协议，不能与前面开发曲线的 GPU 分数混用。')
 else:p('本对照尚未完整完成：'+str((study.get('status') or {}).get('stage','unknown'))+'。不将部分训练或部分评价替代完整对照结果。')
 p('物理轨迹来自当前仿真模型，仍可能含命令偏差或不自然动作；执行成功不构成语义准确的保证。筛选会偏向教师已经能执行的动作，只能作为后续统一物理参考训练的初步测试，不能等价为 OmniTrack 的完整复现。')
'''
assert needle in s;s=s.replace(needle,block+needle)
needle=" table(qrows,[111,94,138,152]);"
replacement=""" paired=data['table_quality'].get('paired_common_completed',{}).get('stable')
 if paired:
  key='walk_torso_angular_velocity_highpass5hz_rms';p(f'在两者都完成的相同 {paired["requests"]} 个场景上，本轮 / 旧稳定版的步行躯干高频 RMS 为 {paired["candidate"][key]:.4f} / {paired["baseline"][key]:.4f} rad/s。这个配对子集仍以完成为条件。','small')
 table(qrows,[111,94,138,152]);"""
assert needle in s;s=s.replace(needle,replacement);p.write_text(s)
