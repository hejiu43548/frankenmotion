"""CPU check: unequal reset frequency cannot erase task balance in fixed slots."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from unified_motion_20261004 import MultiMotionCommand,MotionCommand
U=Path('/home/pku/frankenmotion/outputs_amass/franken_unified_20261004');metadata=U/'training_corpus_v2/clips.json';rows=json.loads(metadata.read_text())
def stub_parent(self,cfg,env):
    self.cfg=cfg;self._env=env;self.motion=SimpleNamespace(time_step_total=rows['ends'][-1]);self.time_steps=torch.zeros(env.num_envs,dtype=torch.long)
    self.metrics={k:torch.zeros(env.num_envs) for k in ['sampling_entropy','sampling_top1_prob']}
torch.manual_seed(7114);env=SimpleNamespace(device='cpu',num_envs=132)
with patch.object(MotionCommand,'__init__',stub_parent):
    command=MultiMotionCommand(SimpleNamespace(clip_metadata=str(metadata),start_probability=.35,task_balanced_slots=True),env)
    command._uniform_sampling(torch.arange(132));visits=torch.zeros(11,dtype=torch.long)
    for i in range(1000):
        # Some environments reset at every step, others only once per 50 steps.
        ids=torch.arange(132) if i%50==0 else torch.arange(12)
        command._uniform_sampling(ids)
        assert torch.equal(command.clip_task[command.clip_ids],command.env_task)
        assert torch.all(command.time_steps>=command.starts[command.clip_ids])
        assert torch.all(command.time_steps<command.ends[command.clip_ids]-1)
        visits+=torch.bincount(command.clip_task[command.clip_ids],minlength=11)
assert visits.tolist()==[12000]*11
report=dict(scope='CPU sampler logic; simulator integration still requires a smoke run',unequal_reset_iterations=1000,environment_slots=132,tasks=command.task_names,task_observation_assignments=visits.tolist(),all_phase_indices_within_clip=True)
(U/'task_balanced_sampler_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
