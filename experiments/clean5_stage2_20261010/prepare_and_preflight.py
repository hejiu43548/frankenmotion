import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

code_root = Path('/mnt/sda2/frankenmotion/code/clean5_stage2_0344b7c')
sys.path[:0] = [str(code_root), '/tmp/frankenmotion_hydra_deps']
os.chdir(code_root)
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from shared_motion.training.data import MotionDataset, assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.catalog import COMMAND_RANGES, TASK_NAMES, measure

output = Path('/mnt/sda2/frankenmotion/outputs_amass/clean5_stage2_20261010')
output.mkdir(exist_ok=False)
data_directory = output / 'data'
data_directory.mkdir()
source = Path('/mnt/sda2/frankenmotion/outputs_amass/source_repair_sidestep_20261010_v2')
tasks = ['wave', 'strike', 'kick', 'sidestep', 'turn']
expected_counts = {'train': {'wave':45,'strike':13,'kick':33,'sidestep':34,'turn':591}, 'val': {'wave':3,'strike':1,'kick':2,'sidestep':6,'turn':84}}
expected_hashes = {'train':'89551a8221c7c796cce2df14a6eafa5bec24ac3322434945db45fb924d20e8a3','val':'4ae872d8f48e2ca8579f188869f21f271c4a9e8c09a7b742a52e756e43122928'}
provenance = {'source_directory':str(source), 'source_manifest_sha256':{}, 'selected_manifest_sha256':{}, 'counts':{}, 'tasks':tasks, 'user_authorization':'User confirmed starting Stage 2 with the five repaired tasks reviewed in this chat, 716 train / 96 val.', 'stage3_enabled':False}
for split in ['train','val']:
    content = (source / (split+'.json')).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    assert digest == expected_hashes[split], 'Reviewed source manifest changed'
    selected = [record for record in json.loads(content) if record['task'] in tasks]
    assert dict(Counter(record['task'] for record in selected)) == expected_counts[split]
    assert all(record['split']==split for record in selected)
    target = data_directory / (split+'.json')
    target.write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
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

overrides = ['stage=stage2','controller=with_root','experiment=clean5','run_date=20261010','seed=20261011',f'initial={root_path}',f'stage.initial={root_path}',f'backbone.checkpoint={backbone_path}',f'data.train_manifest={data_directory}/train.json',f'data.val_manifest={data_directory}/val.json','data.path_root=/home/psirobot/projects/frankenmotion',f'data.skeleton={skeleton_path}','data.tasks=[wave,strike,kick,sidestep,turn]','stage.steps=70000','stage.batch_size=40','stage.eval_every=1000','validation.points=10','validation.ddim_steps=50',f'paths.report={output}/reports',f'paths.artifacts={output}/artifacts']
with initialize_config_dir(version_base='1.3',config_dir=str(code_root/'config')):
    config = compose(config_name='train',overrides=overrides)
OmegaConf.save(config,output/'planned_config.yaml',resolve=True)
command = ['/home/psirobot/projects/frankenmotion/.venv_unified/bin/python','-u',str(code_root/'scripts/train.py'),*overrides]
launch = {'command':command,'cwd':str(code_root),'environment':{'PYTHONPATH':'/tmp/frankenmotion_hydra_deps:'+str(code_root)},'stage':2,'stage3_enabled':False,'state':'preflight_passed','output':str(output)}
(output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'preflight':provenance['preflight'],'counts':provenance['counts'],'output':str(output)},indent=2),flush=True)
