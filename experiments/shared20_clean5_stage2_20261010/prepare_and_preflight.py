import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

code_root = Path('/mnt/sda2/frankenmotion/code/shared20_clean5_stage2_0344b7c')
sys.path[:0] = [str(code_root), '/tmp/frankenmotion_hydra_deps']
os.chdir(code_root)
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from shared_motion.training.data import MotionDataset, assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.catalog import COMMAND_RANGES, TASK_NAMES, measure

output = Path('/mnt/sda2/frankenmotion/outputs_amass/shared20_clean5_stage2_20261010')
output.mkdir(exist_ok=False)
data_directory = output / 'data'
data_directory.mkdir()
source = Path('/mnt/sda2/frankenmotion/outputs_amass/source_repair_sidestep_20261010_v2')
tasks = ['raise_hand', 'reach', 'strike', 'wave', 'turn', 'sidestep', 'back_walk', 'kick', 'jump', 'lean', 'walk', 'squat', 'bow', 'clap', 'point', 'stretch', 'twist', 'march', 'jog', 'arm_circle']
expected_counts = {'train': {'raise_hand': 940, 'reach': 301, 'strike': 13, 'wave': 45, 'turn': 591, 'sidestep': 34, 'back_walk': 1118, 'kick': 33, 'jump': 759, 'lean': 346, 'walk': 3868, 'squat': 313, 'bow': 95, 'clap': 151, 'point': 115, 'stretch': 779, 'twist': 184, 'march': 42, 'jog': 742, 'arm_circle': 428}, 'val': {'raise_hand': 122, 'reach': 47, 'strike': 1, 'wave': 3, 'turn': 84, 'sidestep': 6, 'back_walk': 135, 'kick': 2, 'jump': 94, 'lean': 49, 'walk': 469, 'squat': 44, 'bow': 30, 'clap': 20, 'point': 10, 'stretch': 74, 'twist': 26, 'march': 2, 'jog': 92, 'arm_circle': 61}}
expected_hashes = {'train':'89551a8221c7c796cce2df14a6eafa5bec24ac3322434945db45fb924d20e8a3','val':'4ae872d8f48e2ca8579f188869f21f271c4a9e8c09a7b742a52e756e43122928'}
provenance = {'source_directory':str(source), 'source_manifest_sha256':{}, 'selected_manifest_sha256':{}, 'counts':{}, 'tasks':tasks, 'user_authorization':'User explicitly corrected scope: train ALL 20 tasks, replacing five tasks with repaired data; 10897 train / 1371 val. This new run starts TaskControl from initialization and does not resume the erroneous five-task run.', 'stage3_enabled':False}
for split in ['train','val']:
    content = (source / (split+'.json')).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    assert digest == expected_hashes[split], 'Reviewed source manifest changed'
    selected = json.loads(content)
    assert len(selected) == sum(expected_counts[split].values())
    baseline = json.loads((source.parent / 'shared20_walking_data_20261009' / (split+'.json')).read_text())
    repaired = {'wave','strike','kick','sidestep','turn'}
    assert [record for record in selected if record['task'] not in repaired] == [record for record in baseline if record['task'] not in repaired], 'Other 15 tasks changed'
    assert dict(Counter(record['task'] for record in selected)) == expected_counts[split]
    assert all(record['split']==split for record in selected)
    target = data_directory / (split+'.json')
    target.write_bytes(content)
    provenance['source_manifest_sha256'][split] = digest
    provenance['selected_manifest_sha256'][split] = hashlib.sha256(target.read_bytes()).hexdigest()
    provenance['counts'][split] = expected_counts[split]

skeleton_path = '/home/psirobot/projects/frankenmotion/outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz'
root_path = '/mnt/sda2/frankenmotion/outputs_amass/shared20_walking_20261009/imported_root.pt'
backbone_path = '/home/psirobot/checkpoints/frankenmotion/frankenmotion.ckpt'
for name,path,expected in [('skeleton',skeleton_path,'f6454fbe22bd135f7de84af8a02a9e633a2a971e7dbc5f03ba577668735e15b9'),('root',root_path,'ec279c4d78f0a5c8355f4124a4a6224731abbedfca08dceac9e1dfec5fd1e4e6'),('backbone',backbone_path,'c9dca1988dd08dd9e2164ac4cf6ae8fece23011a6371e83d502cdfabe5352e18')]:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest == expected, name+' hash mismatch'
    provenance[name+'_sha256'] = digest

assert tuple(COMMAND_RANGES[TASK_NAMES.index('sidestep')]) == (0.4,1.0)
torch.set_num_threads(4)
assert torch.cuda.is_available()
skeleton = Skeleton(skeleton_path).cuda()
datasets = {split:MotionDataset(data_directory/(split+'.json'),split,tasks,'/home/psirobot/projects/frankenmotion') for split in ['train','val']}
assert_disjoint(datasets['train'],datasets['val'])
checks = {}
with torch.no_grad():
    for split,dataset in datasets.items():
        max_error = 0.0
        for record in dataset.rows:
            if record.get('cache_sha256'):
                assert record['cache_sha256'] == dataset.cache_hashes[record['cache']]
        for offset in range(0,len(dataset.rows),20):
            batch = dataset.batch(list(range(offset,min(offset+20,len(dataset.rows)))),'cuda')
            quantities = measure(skeleton,batch['motion'],batch['task'],batch['lengths'])
            errors = (quantities-batch['quantity'].reshape_as(quantities)).abs()
            assert torch.isfinite(errors).all()
            max_error = max(max_error,float(errors.max()))
            for record_index in range(offset,min(offset+20,len(dataset.rows))):
                item_quantity=float(dataset.items[record_index]['quantity'])
                assert abs(item_quantity-dataset.rows[record_index]['quantity']) < 1e-5
        assert max_error < 1e-4, (split,max_error)
        checks[split] = {'records':len(dataset.rows),'cache_fingerprint':dataset.fingerprint,'fk_quantity_max_error':max_error}
provenance['preflight'] = {'checks':checks,'train_val_family_disjoint':True,'sidestep_range':[0.4,1.0],'gpu':torch.cuda.get_device_name(0)}
(data_directory/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')

overrides = ['stage=stage2','controller=with_root','experiment=shared20_clean5','run_date=20261010','seed=20261011',f'initial={root_path}',f'stage.initial={root_path}',f'backbone.checkpoint={backbone_path}',f'data.train_manifest={data_directory}/train.json',f'data.val_manifest={data_directory}/val.json','data.path_root=/home/psirobot/projects/frankenmotion',f'data.skeleton={skeleton_path}','data.tasks=[raise_hand,reach,strike,wave,turn,sidestep,back_walk,kick,jump,lean,walk,squat,bow,clap,point,stretch,twist,march,jog,arm_circle]','stage.steps=70000','stage.batch_size=40','stage.eval_every=1000','validation.points=10','validation.ddim_steps=50',f'paths.report={output}/reports',f'paths.artifacts={output}/artifacts']
with initialize_config_dir(version_base='1.3',config_dir=str(code_root/'config')):
    config = compose(config_name='train',overrides=overrides)
OmegaConf.save(config,output/'planned_config.yaml',resolve=True)
command = ['/home/psirobot/projects/frankenmotion/.venv_unified/bin/python','-u',str(code_root/'scripts/train.py'),*overrides]
launch = {'command':command,'cwd':str(code_root),'environment':{'PYTHONPATH':'/tmp/frankenmotion_hydra_deps:'+str(code_root)},'stage':2,'stage3_enabled':False,'state':'preflight_passed','output':str(output)}
(output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'preflight':provenance['preflight'],'counts':provenance['counts'],'output':str(output)},indent=2),flush=True)
