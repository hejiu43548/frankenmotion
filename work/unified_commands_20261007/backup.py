from pathlib import Path
import json,shutil,hashlib,time
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';D.mkdir(exist_ok=False);W.mkdir(exist_ok=True);B=D/'backup';B.mkdir()
paths={'current_full':R/'outputs_amass/unified_generator_20261007/exports/selected_final/unified_generator.pt','root':R/'outputs_amass/root_control_20260925/best.pt','physical':R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt','goal':R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt','reach':R/'outputs_amass/reach_demo_20261005/frozen/reach_adapter.pt','exit':R/'outputs_amass/reach_demo_20261005/frozen/exit_adapter.pt','tracker':R/'outputs_amass/unified_generator_20261007/backup/tracker_candidate_actor.pt','tracker_base':R/'outputs_amass/unified_generator_20261007/backup/tracker_actor.pt','table_tracker':R/'outputs_amass/turn_demo_20261005/frozen/actor.pt'}
rows=[]
for key,src in paths.items():
 dst=B/(key+'.pt');shutil.copy2(src,dst);h=hashlib.sha256(src.read_bytes()).hexdigest();assert h==hashlib.sha256(dst.read_bytes()).hexdigest();rows.append(dict(name=key,source=str(src),backup=str(dst),sha256=h,bytes=dst.stat().st_size))
 if src.with_suffix('.json').exists():shutil.copy2(src.with_suffix('.json'),dst.with_suffix('.json'))
code=B/'code';code.mkdir()
for p in [R/'work/table_goal_adapter_20261005.py',R/'work/reach_adapter_v6_20261005.py',R/'work/reach_exit_adapter_v6_20261005.py',R/'work/physical_adapter_20261003.py',R/'outputs_amass/root_control_20260925/code/control.py',R/'outputs_amass/franken_eleven_20261003/code/core.py']:
 shutil.copy2(p,code/p.name)
shutil.copytree(R/'work/unified_generator_20261007',code/'previous_unified',ignore=shutil.ignore_patterns('__pycache__'))
(D/'backup_manifest.json').write_text(json.dumps(dict(created=time.time(),files=rows),indent=2));print(json.dumps(rows,indent=2))
