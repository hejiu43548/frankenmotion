import argparse,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--name',required=True);a=p.parse_args();folder=D/a.folder;out=[]
for row in json.loads((folder/'manifest.json').read_text()):
 run=Path(row['source'])/a.name
 if not (run/'result.json').exists():continue
 r=json.loads((run/'result.json').read_text());item={'index':row['index'],'success':r['success'],'command':r['command_metrics']}
 for seg in ['walk','exit']:
  subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/reach_gait_metrics_20261005.py'),'--run',str(run),'--segment',seg],stdout=subprocess.DEVNULL,check=True)
  item[seg]=json.loads((run/(seg+'_gait_metrics.json')).read_text())
 out.append(item)
(folder/(a.name+'_quality.json')).write_text(json.dumps(out,indent=2))
for r in out:
 print(r['index'],r['success'],r['command']['actual_hold_wrist_forward_human_equiv_m'],{s:{k:r[s].get(k) for k in ['clearance_p95_m','mean_contact_point_slip_m_s','knee_std_deg']} for s in ['walk','exit']},flush=True)
