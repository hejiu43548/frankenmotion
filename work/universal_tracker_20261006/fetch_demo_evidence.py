from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';remote='/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006';runs=[]
pair=json.loads((D/'visuals/frozen_candidates/candidate_index.json').read_text())['candidates'][0]
runs += [r['run'] for r in pair['rows']]
for name in ['service_trimmed','navigation','retreat','long_horizon_v3','long_no_anchors']:
 runs += [r['run'] for r in json.loads((D/f'general_evaluation/frozen_demo_{name}/results.json').read_text())]
runs += [r['run'] for r in json.loads((D/'visuals/eleven_grid/protocol.json').read_text())['records']]
runs += [remote+'/general_evaluation/frozen_demo_point/point_01584_s0']
runs += [r['run'] for r in json.loads((D/'general_evaluation/long_horizon_delay_20ms/results.json').read_text())]
paths=[]
for run in sorted(set(runs)):
 relative=Path(run).relative_to(remote)
 for name in ['actual.npz','motion.npz','initial_motion.npz','inference_contract.json','result.json']:
  paths.append(str(relative/name))
p=D/'demo_evidence_files.txt';p.write_text('\n'.join(paths)+'\n');subprocess.run(['rsync','-az','--files-from='+str(p),'pku@100.94.2.67:'+remote+'/',str(D)+'/'],check=True);print('Fetched',len(set(runs)),'raw demo trajectories and references')
