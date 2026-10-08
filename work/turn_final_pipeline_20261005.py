import subprocess,json,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';gen=R/'.conda/bin/python';gmr=R/'work/g1_sim_env/bin/python';sim=R/'work/mjlab_stable_env/bin/python'
def run(py,script,*args):
 print('START',script,*args,flush=True);subprocess.run([str(py),str(R/'work'/script),*map(str,args)],cwd=R,check=True);print('DONE',script,flush=True)
run(gen,'turn_generate_walk_20261005.py','--weight',D/'frozen/goal_adapter.pt','--name','final_walk','--count',6,'--seed',91005000,'--all-random','--refine-iterations',3)
run(gmr,'reach_prepare_walk_20261005.py','--folder',D/'final_walk')
run(gen,'turn_generate_reach_20261005.py','--name','final_reach','--walk-folder',D/'final_walk','--seed',92005000)
run(gmr,'reach_prepare_corpus_v3_20261005.py','--folder',D/'final_reach')
run(gen,'turn_generate_walk_20261005.py','--weight',D/'frozen/goal_adapter.pt','--name','final_away','--count',6,'--seed',94005000,'--layout-json',D/'final_away_layouts.json','--refine-iterations',3)
run(gmr,'reach_prepare_walk_20261005.py','--folder',D/'final_away')
run(gen,'turn_generate_turn_final_20261005.py')
run(gmr,'turn_compose_final_20261005.py','--name','final12')
run(sim,'turn_convert_20261005.py','--folder',D/'final12')
run(sim,'turn_native_batch_20261005.py','--folder',D/'final12','--name','selected','--checkpoint',D/'frozen/policy.pt','--actor',D/'frozen/actor.pt')
(D/'final_pipeline_complete.json').write_text(json.dumps(dict(completed_unix=time.time())))
