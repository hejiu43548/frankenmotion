from pathlib import Path
import sys,json,hashlib
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/temporal_mix_20261008';sys.path[:0]=[str(R/'work/unified_commands_20261007'),str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code'),str(R)]
from single_runtime import load_one_checkpoint
from shared_infer import generate,command_features
from command_inputs import encode_control
from core import FK,torch,np,HH
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.tools.parse_user_input import parse_and_validate_user_input
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align
import src.prepare
torch.set_num_threads(3);name='temporal_mix_emphasis_20261008';out=R/'outputs_amass/unified_commands_20261007/generation'/name;out.mkdir(exist_ok=False);fk=FK('cpu')
def a(text,start,end):return dict(text=text,start=start,end=end)
walk=[a('stand still',0,1),a('walk forward',1,5),a('stop walking and stand',5,6)]
parts=dict(action=walk,trajectory=walk,spine=[a('upright',0,6)],head=[a('look forward',0,6)],left_leg=walk,right_leg=walk,left_arm=[a('arm relaxed',0,2),a('raise left hand and wave hello repeatedly',2,4),a('lower left hand and relax the left arm',4,6)],right_arm=[a('arm relaxed',0,3),a('raise right hand and wave hello repeatedly',3,5),a('lower right hand and relax the right arm',5,6)])
parts['action']=[a('stand still',0,1),a('walk forward',1,2),a('walk forward while waving hello with the left hand',2,3),a('walk forward while waving hello with both hands',3,4),a('walk forward while waving hello with the right hand',4,5),a('stop walking and lower both arms',5,6)]
prompt=dict(duration=6.,body_parts=parts,sequence_caption='A person walks forward while raising their arms and waving hello, first with the left hand, then both hands, then the right hand.')
f=D/'prompt_emphasis.json';f.write_text(json.dumps(prompt,indent=2));cfg=OmegaConf.load(R/'outputs_amass/root_control_20260925/base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False
ann=parse_and_validate_user_input(str(f),cfg=cfg,fps=20);emb,_=load_from_annotation_with_model(enc,ann['annotations'],ann['path'],ann['start'],ann['end']);local=align(torch.zeros(120,205),emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])};torch.save(dict(local=local,tx=tx),D/'prompt_emphasis.pt');del enc
ck=D/'backup/generator.pt';m=load_one_checkpoint(ck);base=command_features('root_profile',0.,120,[0,0,0,0,1,1]);t=torch.arange(120)*.05;envelope=torch.clamp((t-1)/.2,0,1)*torch.clamp((5-t)/.2,0,1);values=torch.zeros(1,120,2);values[0,:,0]=.6*fk.height/HH*envelope;base[:,:,18:22]=encode_control(values,torch.ones(1,120,dtype=torch.bool));rows=[]
for mode in ['text_only','speed_0p6']:
 c=base.clone()
 if mode=='text_only':c[:,:,17]=0;c[:,:,20:22]=0;c[:,:,25]=0;c[:,:,27]=0
 for si in range(1):
  seed=10808100+si;raw=generate(m,local,tx,c,seed);pos,poses,root=fk(raw,canonical=False,return_pose=True);path=out/f'{mode}_s{si}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.,task='walk',command=.6,control_features=c[0].numpy())
  rows.append(dict(kind='temporal_composition',task='walk',source=f'{mode}_s{si}',seed=seed,command=.6,command_index=si,path=str(path),generator=str(ck),generator_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),mode=mode,split='exploratory',scope='6s timeline:walk1-5,leftwave2-4,rightwave3-5. Text_only has no numeric constraint; speed_0p6 uses human-equivalent0.6m/s with0.2s boundary ramps. No wave amplitude scalar. Not standard11 benchmark.'))
  print(mode,si,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2));(out/'protocol.json').write_text(json.dumps(dict(split='exploratory',prompt=prompt,weight_sha256=rows[0]['generator_sha256'],training=False,numerical_speed_human_equiv=.6,transition_ramp_s=.2,seed_base=10808100,cases=2),indent=2));(D/'emphasis_generation_complete.json').write_text(json.dumps(dict(name=name,requests=2)))
