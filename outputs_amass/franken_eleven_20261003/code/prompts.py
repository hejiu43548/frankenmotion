from core import *
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.tools.parse_user_input import parse_and_validate_user_input
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align

CAPTIONS={
'raise_hand':['A person raises their right hand above the shoulder.','A standing person lifts their right arm upward.','A person lifts the right hand high and holds it up.','A person stands upright and raises the right hand.'],
'reach':['A person reaches straight forward with the right hand.','A standing person extends their right arm forward.','A person stretches the right hand out in front.','A person reaches out in front of their body with the right arm.'],
'strike':['A person throws a straight right punch and retracts it.','A person punches forward with the right fist then pulls it back.','A standing person performs one straight right-hand punch.','A person jabs straight ahead with their right arm.'],
'wave':['A person waves their right hand repeatedly.','A standing person waves hello with their right hand.','A person raises their right hand and waves from side to side.','A person stands upright and repeatedly waves the right hand.'],
'turn':['A person turns to the right in place.','A standing person rotates their body clockwise.','A person steps around to face right.','A person makes a right turn while staying near the same spot.'],
'sidestep':['A person steps sideways to the right.','A person moves to the right with sideways steps.','A standing person sidesteps right without turning.','A person takes several side steps to the right.'],
'back_walk':['A person walks backwards.','A person takes steady steps backwards.','A person walks backward while facing forward.','A standing person begins walking backwards.'],
'kick':['A person kicks forward with the right foot then lowers it.','A person makes a forward kick with the right leg.','A standing person kicks the right foot straight ahead.','A person raises and kicks their right leg forward then returns it.'],
'jump':['A person jumps up with both feet and lands.','A standing person jumps vertically and lands on both feet.','A person bends their knees and jumps straight up.','A person performs a two-foot vertical jump and lands.'],
'lean':['A person leans forward and holds the posture.','A standing person bends the torso forward and remains there.','A person tilts their upper body forward and holds.','A person stands with feet planted and leans forward.'],
'walk':['A person walks forward.','A person takes steady steps forward.','A person walks straight ahead.','A standing person begins walking forward.']}

def parts(task,d):
    def a(text,start=0,end=None):return dict(text=text,start=start,end=d if end is None else end)
    out={'head':[a('look forward')],'spine':[a('upright')],'left_leg':[a('stand')],'right_leg':[a('stand')],'left_arm':[a('relaxed')],'trajectory':[a('stand still')]}
    if task in ['raise_hand','reach']:
        verb='raise right hand upward' if task=='raise_hand' else 'reach forward'
        out.update(action=[a(verb)],right_arm=[a('relaxed',0,.3),a(verb,.3,1.2),a('hold arm position',1.2,d)])
    elif task=='strike':out.update(action=[a('punch forward')],right_arm=[a('ready',0,.6),a('punch straight forward',.6,1.5),a('retract arm',1.5,d)])
    elif task=='wave':out.update(action=[a('wave hello')],right_arm=[a('raise hand',0,.6),a('wave repeatedly',.6,5.2),a('lower hand',5.2,d)])
    elif task in ['walk','back_walk','sidestep','turn']:
        verb={'walk':'walk forward','back_walk':'walk backward','sidestep':'step sideways right','turn':'turn right'}[task]
        out.update(action=[a(verb)],trajectory=[a(verb)],left_leg=[a(verb)],right_leg=[a(verb)],right_arm=[a('swing naturally')],left_arm=[a('swing naturally')])
    elif task=='kick':out.update(action=[a('kick right foot forward')],right_leg=[a('stand',0,.4),a('kick forward',.4,1.7),a('lower foot',1.7,d)],right_arm=[a('balance')])
    elif task=='jump':
        leg=[a('bend knees',0,.7),a('jump up',.7,1.7),a('land',1.7,d)];out.update(action=[a('jump up and land')],trajectory=[a('jump vertically')],left_leg=leg,right_leg=leg,right_arm=[a('balance')])
    elif task=='lean':out.update(action=[a('lean forward and hold')],spine=[a('upright',0,.4),a('lean forward',.4,1.0),a('hold forward lean',1.0,d)],right_arm=[a('relaxed')])
    return out

def main():
    torch.set_num_threads(2);cfg=OmegaConf.load(OLD/'base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False
    (OUT/'prompts').mkdir(exist_ok=True);manifest=[]
    for task in TASKS:
        tid=TASKS.index(task);n=FRAMES[tid];d=n/20
        for pi,caption in enumerate(CAPTIONS[task]):
            source=dict(duration=d,sequence_caption=caption,body_parts=parts(task,d));path=OUT/'prompts'/f'{task}_p{pi}.json';path.write_text(json.dumps(source,indent=2))
            ann=parse_and_validate_user_input(str(path),cfg=cfg,fps=20);emb,_=load_from_annotation_with_model(enc,ann['annotations'],ann['path'],ann['start'],ann['end'])
            motion=torch.zeros(n,205);local=align(motion,emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])}
            torch.save(dict(local=local,tx=tx),path.with_suffix('.pt'))
            for si in range(4):
                seed=26010300+pi*100+si;manifest.append(dict(task=task,task_id=tid,source=f'{task}_p{pi}_s{si}',prompt=str(path),caption=caption,seed=seed,frames=n,commands=np.linspace(*RANGES[tid],5).tolist()))
    save('evaluation_manifest.json',manifest)
    save('evaluation_protocol.json',dict(tasks=TASKS,ranges=RANGES,tolerances=TOLS,frames=FRAMES,units=UNITS,sources_per_task=16,commands_per_source=5,source_definition='4 fixed text prompts x 4 diffusion seeds. Each pair shares noise across five commands. Not the collaborator source set.',walk_quantity='legacy XY path speed for screenshot comparability; net forward speed also reported',human_equivalent_height=HH,baseline='existing frozen base plus old root adapter; numeric inputs supported only for walk/back_walk/turn',new_arm='new task adapter injected during denoising, no output pose editing or per-test simulation fitting',sampler='50-step deterministic DDIM, guidance 1',failures='all planned requests retained; missing actual quantity charged 1 in normalized E_all; no extrapolation beyond falls'))
    print('Frozen',len(manifest),'prompt/noise sources')
if __name__=='__main__':main()
