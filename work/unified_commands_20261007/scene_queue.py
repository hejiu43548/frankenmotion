from pathlib import Path
import subprocess,argparse
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';py=R/'.conda/bin/python';ap=R/'work/mjlab_stable_env/bin/python';actor=D/'backup/tracker.pt'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint');p.add_argument('--full',action='store_true');p.add_argument('--layouts',type=int,default=4);p.add_argument('--seed',type=int,default=107086000);a=p.parse_args();name=a.name;folder=D/'table'/name
steps=[(py,[W/'generate_scenes.py','--name',name]+(['--checkpoint',a.checkpoint] if a.checkpoint else ['--teachers'])+(['--full'] if a.full else [])+['--layouts',a.layouts,'--seed',a.seed]),(ap,[R/'work/reach_prepare_walk_20261005.py','--folder',folder/'walks']),(ap,[R/'work/reach_prepare_walk_20261005.py','--folder',folder/'away']),(ap,[R/'work/reach_prepare_corpus_v3_20261005.py','--folder',folder/'reaches']),(ap,[W/'compose_scenes.py','--name',name]),(ap,[R/'work/turn_convert_20261005.py','--folder',folder/'composed']),(ap,[W/'scene_tracking.py','--folder',folder/'composed','--name','shared_tracker','--actor',actor,'--checkpoint',actor])]
for i,(exe,args) in enumerate(steps):
 with (D/f'{name}_scene_{i}.log').open('w') as f:subprocess.run([str(x) for x in [exe]+args],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
