from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';out=D/'tables';out.mkdir(exist_ok=True);data=json.loads((D/'report_data.json').read_text());tol=dict(raise_hand=.02,reach=.02,strike=.1,wave=.015,turn=.08,sidestep=.06,back_walk=.05,kick=.05,jump=.04,lean=.06,walk=.05)
def write(name,rows):
 assert rows
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for tag in ['candidate','broad','stable','sonic']:
 path=D/(f'general_evaluation/fresh_{tag}/audited_results.json' if tag!='sonic' else 'sonic_evaluation/fresh_final/audited_results.json')
 for r in json.loads(path.read_text()):
  row=dict(controller=tag,task=r['task'],source=r['source'],seed=r['seed'],command_index=r['command_index'],command=r['command'],tolerance=tol[r['task']],physical_complete=r['physical_complete'])
  for stage in ['human','g1','actual']:
   v=r.get(stage);row[stage+'_quantity']=v['quantity'] if v else None;row[stage+'_event']=v['event_pass'] if v else False;row[stage+'_joint_pass']=bool(v and v['event_pass'] and abs(v['quantity']-r['command'])<=tol[r['task']])
  row['run']=r.get('run');rows.append(row)
write('fresh_880_all_controllers.csv',rows)
rows=[]
for tag,c in data['controllers'].items():
 for task,r in c['per_task'].items():rows.append(dict(controller=tag,task=task,**r['actual'],class_gate=c['class_pass'][task]))
write('per_class_summary.csv',rows)
write('controller_summary.csv',[dict(controller=tag,**c['aggregate']) for tag,c in data['controllers'].items()])
rows=[]
for tag,c in data['controllers'].items():
 for task,r in c['natural']['results'].items():rows.append(dict(controller=tag,task=task,**r))
write('natural_test_summary.csv',rows)
write('robustness_summary.csv',[dict(controller=tag,profile=profile,**r['aggregate']) for tag,rr in data['robustness'].items() for profile,r in rr.items()])
if data.get('replication',{}).get('results'):
 r=data['replication']['results'];write('second_seed_per_class.csv',[dict(task=t,**v['actual']) for t,v in r['per_task'].items()])
training=[]
for p in sorted((D/'training').glob('*/protocol.json')):
 protocol=json.loads(p.read_text());args=protocol.get('arguments',{});done=p.parent/'complete.json';completion=json.loads(done.read_text()) if done.exists() else {};training.append(dict(name=p.parent.name,complete=done.exists(),requested_updates=args.get('steps'),environments=args.get('envs'),seed=args.get('seed'),dataset=args.get('dataset'),initial=args.get('initial'),learning_rate=args.get('learning_rate'),long_preview=args.get('long_preview'),physical_smoothness=args.get('physical_smoothness'),retention_weight=args.get('retention_weight'),wall_seconds=completion.get('wall_s')))
write('training_inventory.csv',training)
(out/'README.txt').write_text('空白 quantity 表示不可测，不是零。joint_pass 要求事件与数值容差同时满足。SONIC 使用独立原生模型/PD系统协议，不能视为等执行器网络消融。置信区间、详细协议与SHA256见上级report_data.json。所有单位及事件定义见 ../release/metric_definitions.json。\n',encoding='utf8')
print('Exported evidence tables')
