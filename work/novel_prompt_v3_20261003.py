"""Frozen V3 paraphrase stress test; same task schedules, new text strings and noise."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import physical_adapter_20261003 as pa
from core import *
from prompts import parts,CAPTIONS
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.tools.parse_user_input import parse_and_validate_user_input
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align
from generate import inputs
NEW=pa.OUT;folder=NEW/'novel_prompt_v3';folder.mkdir(exist_ok=True);(folder/'prompts').mkdir(exist_ok=True);(folder/'generated').mkdir(exist_ok=True)
CAPS=['While standing, bring the right hand up overhead.','Keeping your stance, extend the right hand directly in front of you.','Deliver one straight jab with the right fist, then draw the arm back.','Greet someone by repeatedly moving the raised right hand from side to side.','Use small steps to rotate clockwise on the spot.','Move laterally toward your right while keeping your chest facing ahead.','Take a series of steps in reverse without turning around.','Swing the right foot forward for a kick and then return it to the ground.','Push off vertically with both legs and come back down onto both feet.','Keep the feet still, angle the trunk forward, and maintain that position.','Proceed straight ahead with a steady walking gait.']
REPHRASE={'look forward':'keep looking straight ahead','upright':'keep the torso erect','stand':'remain standing','relaxed':'rest the arm naturally','stand still':'stay in one place','raise right hand upward':'lift the right arm toward the ceiling','reach forward':'extend the arm straight ahead','hold arm position':'maintain the extended arm pose','punch forward':'jab straight ahead','ready':'prepare the arm','punch straight forward':'deliver a direct forward jab','retract arm':'draw the arm back','wave hello':'greet with a hand gesture','raise hand':'bring the hand up','wave repeatedly':'move the hand from side to side repeatedly','lower hand':'bring the hand down','walk forward':'take steps straight ahead','walk backward':'take steps in reverse','step sideways right':'travel laterally to the right','turn right':'rotate clockwise in place','swing naturally':'let the arm swing with the steps','kick right foot forward':'swing the right foot into a forward kick','kick forward':'extend the leg with a forward kick','lower foot':'place the foot back down','balance':'use the arm to maintain balance','bend knees':'flex the knees','jump up':'push off the ground vertically','land':'come down onto the feet','jump up and land':'leap upward and return to the ground','jump vertically':'move straight up and back down','lean forward and hold':'incline the trunk forward and keep it there','lean forward':'angle the torso toward the front','hold forward lean':'maintain the forward-inclined posture'}
protocol=dict(status='frozen before inference',scope='11 held-out paraphrases relative to task-adapter prompt set; unchanged task labels/body-part schedules; 2 fresh noises x5commands each; not unseen-action or arbitrary-text generalization',noise='75028000+task_id*100+seed_index',captions=dict(zip(TASKS,CAPS)),phrase_mapping=REPHRASE,integrated_protocol=str(NEW/'frozen_integrated_v3/protocol.json'))
p=folder/'protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol,'Frozen test protocol mismatch'
else:p.write_text(json.dumps(protocol,indent=2))
assert not (folder/'manifest.json').exists(),'Completed test already exists'
torch.set_num_threads(2)
cfg=OmegaConf.load(OLD/'base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False;manifest=[]
for tid,task in enumerate(TASKS):
 assert CAPS[tid] not in CAPTIONS[task];n=FRAMES[tid];body=parts(task,n/20)
 for events in body.values():
  for e in events:e['text']=REPHRASE.get(e['text'],e['text'])
 source=dict(duration=n/20,sequence_caption=CAPS[tid],body_parts=body);path=folder/'prompts'/(task+'.json');path.write_text(json.dumps(source,indent=2));ann=parse_and_validate_user_input(str(path),cfg=cfg,fps=20);emb,_=load_from_annotation_with_model(enc,ann['annotations'],ann['path'],ann['start'],ann['end']);local=align(torch.zeros(n,205),emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])};torch.save(dict(local=local,tx=tx),path.with_suffix('.pt'))
 for si in range(2):manifest.append(dict(task=task,task_id=tid,source=f'{task}_novel_p0_s{si}',prompt=str(path),caption=CAPS[tid],seed=75028000+tid*100+si,frames=n,commands=np.linspace(*RANGES[tid],5).tolist()))
del enc;torch.cuda.set_per_process_memory_fraction(.15);fk=FK('cuda');frozen=json.loads((NEW/'frozen_integrated_v3/protocol.json').read_text());results=[];current=None;model=None
for src in manifest:
 weight=NEW/frozen['generation'].get(src['task'],frozen['generation']['all_other_tasks']);assert hashlib.sha256(weight.read_bytes()).hexdigest()==frozen['weight_hashes'][str(weight)]
 if weight!=current:
  if model is not None:del model;torch.cuda.empty_cache()
  model,_=pa.core.load_model(task_weights=weight);current=weight
 local,tx,cmd,controls=inputs(src,src['commands']);raw=sample(model,local,tx,src['task_id'],cmd,[src['seed']]*5,True,root_controls=controls)
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);q=pa.core.quantity(fk(raw),src['task_id'],HH/fk.height)
 for i,c in enumerate(src['commands']):
  path=folder/'generated'/(src['source']+f'_c{i}.npz');np.savez_compressed(path,motion=raw[i].cpu().numpy(),joints_zup_m=pos[i].cpu().numpy(),poses_axisangle=poses[i].cpu().numpy(),root_translation=root[i].cpu().numpy(),human_height=fk.height,human_quantity=float(q[i]),fps=20.,task=src['task'],command=c);results.append(dict(task=src['task'],source=src['source'],seed=src['seed'],command=c,command_index=i,path=str(path),generator=str(weight)))
 print(src['source'],'generated',flush=True)
(folder/'manifest.json').write_text(json.dumps(results,indent=2));print('110 frozen paraphrase requests generated')
