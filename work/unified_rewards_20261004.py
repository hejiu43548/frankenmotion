"""Task-independent joint reference accuracy; same formula for all 29 joints."""
import torch

def joint_reference_accuracy(env, command_name='motion', std=.3,worst_count=0):
    term=env.command_manager.get_term(command_name)
    squared=(term.joint_pos-term.robot_joint_pos).square()
    score=torch.exp(-squared/(std*std))
    if worst_count:
        assert 0<worst_count<=score.shape[-1]
        return .5*score.mean(-1)+.5*score.topk(worst_count,dim=-1,largest=False).values.mean(-1)
    return score.mean(-1)
