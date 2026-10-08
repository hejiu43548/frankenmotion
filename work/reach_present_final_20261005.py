import json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';F=D/'final_paired16';V=D/'videos';V.mkdir(exist_ok=True);rows=json.loads((F/'manifest.json').read_text());valid=[]
for i in range(0,len(rows),2):
 pair=[Path(r['source']) for r in rows[i:i+2]];rr=[json.loads((s/'selected/result.json').read_text()) for s in pair]
 if all(r['success'] for r in rr):
  quality=[json.loads((s/'selected/exit_gait_metrics.json').read_text()) for s in pair];slip=sum(q['mean_contact_point_slip_m_s'] for q in quality)/2;valid.append((rr[0]['scene']['exit_task'],slip,i,pair))
side=min([v for v in valid if v[0]=='sidestep'],key=lambda v:v[1]);back=min([v for v in valid if v[0]=='back_walk'],key=lambda v:v[1]);py=R/'work/mjlab_stable_env/bin/python'
def run(name,args):subprocess.run([str(py),str(R/'work'/f'reach_{name}_20261005.py')]+args,cwd=R,check=True)
selection=dict(side_pair_indices=[side[2],side[2]+1],back_pair_indices=[back[2],back[2]+1],rule='Presentation only: successful near/far pairs, lowest average measured departure contact-point slip within each departure type. All final cases remain in benchmark. Visual inspection still required.');(V/'selection.json').write_text(json.dumps(selection,indent=2))
run('render_pair',['--near',str(side[3][0]/'selected'),'--far',str(side[3][1]/'selected'),'--output',str(V/'reach_030_vs_050.mp4')])
run('render',['--run',str(side[3][1]/'selected'),'--output',str(V/'side_departure_full.mp4')])
run('render',['--run',str(back[3][1]/'selected'),'--output',str(V/'back_departure_full.mp4'),'--label','FrankenMotion | Backward departure prototype'])
for stage in ['walk','exit']:
 for s in [side[3][1],back[3][1]]:run('render_strip',['--run',str(s/'selected'),'--segment',stage])
run('make_movie',['--videos',str(V),'--side',str(V/'side_departure_full.mp4'),'--pair',str(V/'reach_030_vs_050.mp4')]);print(json.dumps(selection),flush=True)
