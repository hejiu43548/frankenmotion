"""Render metric-selected presentation candidates; final visual approval is separate."""
import json,time,subprocess,hashlib,datetime
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python';cutoff=datetime.datetime(2026,10,6,2,10,tzinfo=datetime.timezone.utc).timestamp()
while True:
 p=D/'frozen_test_supervisor_status.json';status=json.loads(p.read_text()) if p.exists() else {}
 if any(status.get(k,{}).get('returncode',0) for k in ['cpu','demos']):raise RuntimeError(status)
 if all(k in status for k in ['cpu','demos']):break
 if time.time()>cutoff:raise RuntimeError('Rendering queue timeout')
 time.sleep(10)
f=json.loads((D/'frozen_unified/protocol.json').read_text());out=D/'visuals/frozen_candidates';out.mkdir(exist_ok=False);inventory=[]
def execute(name,script,args):
 with (out/(name+'.log')).open('w') as log:subprocess.run([str(py),str(script),*map(str,args)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
def verify(row):assert row['checkpoint_sha256']==f['checkpoint_sha256']
def pairs(summary):
 rows=summary['results'];result=[]
 for i in range(0,len(rows),2):
  pair=rows[i:i+2]
  if len(pair)!=2:continue
  for r in pair:verify(r)
  if not all(r['success'] for r in pair):continue
  errors=sum(r['command_metrics']['command_error_m']+r['command_metrics']['turn_error_deg']/100 for r in pair)
  result.append((errors,i,pair))
 return sorted(result,key=lambda x:x[0])
summary=json.loads((D/'table_evaluation/fresh_candidate/summary.json').read_text());fresh=pairs(summary)
if fresh:
 chosen=[('fresh_table',fresh[0])]
else:
 dev=json.loads((D/'table_evaluation'/f['selected']/'summary.json').read_text());ranked=pairs(dev);chosen=[('development_table',ranked[0])] if ranked else []
for label,(_,i,pair) in chosen:
 movies=[]
 for row in pair:
  run=Path(row['run']);stem=label+f'_{row["scene"]["index"]:03d}';movie=out/(stem+'.mp4');movies.append(movie)
  execute(stem,R/'work/turn_render_20261005.py',['--run',run,'--output',movie,'--label','FrankenMotion | Reach, turn & depart'])
  execute(stem+'_walk',W/'render_temporal_strip.py',['--run',run,'--output',out/(stem+'_walk.png'),'--start',1,'--end',4,'--count',20])
 execute(label+'_paired',W/'compose_paired_video.py',['--left',movies[0],'--right',movies[1],'--output',out/(label+'_paired.mp4'),'--title','Same scene and noise | Reach command 0.30 m vs 0.50 m'])
 inventory.append(dict(kind='table_pair',split=label,selection='Both pass full table criteria; minimize summed reach error plus turn error/100. A presentation selection, not a new test success estimate.',rows=pair,video=str(out/(label+'_paired.mp4'))))
for kind in ['navigation','point']:
 base=D/'general_evaluation'/('frozen_demo_'+kind);protocol=json.loads((base/'protocol.json').read_text());assert protocol['checkpoint_sha256']==f['checkpoint_sha256']
 for row in json.loads((base/'results.json').read_text()):
  if not row.get('physical_complete'):continue
  run=Path(row['run']);stem=kind+'_'+run.name
  if kind=='navigation':
   execute(stem,W/'render_navigation_presentation.py',['--run',run,'--output',out/(stem+'.mp4')]);video=out/(stem+'.mp4')
  else:
   execute(stem,W/'render_general.py',['--run',run,'--output',out/stem,'--follow']);video=out/stem/'reference_vs_physics.mp4'
  inventory.append(dict(kind=kind,selection='All physically completed preexisting generated demo candidates rendered; semantic and visual review still required.',row=row,video=str(video)))
(out/'candidate_index.json').write_text(json.dumps(dict(checkpoint_sha256=f['checkpoint_sha256'],scope=__doc__,fresh_table_success=summary['success'],fresh_table_planned=summary['planned'],candidates=inventory,visual_review='pending; no claim that every rendered candidate is a good demo'),indent=2));print('Presentation candidates rendered',len(inventory),flush=True)
