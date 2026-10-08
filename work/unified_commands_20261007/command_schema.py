import torch,math
TASKS=['raise_hand', 'reach', 'strike', 'wave', 'turn', 'sidestep', 'back_walk', 'kick', 'jump', 'lean', 'walk']
RANGES=[(0.35, 0.75), (0.25, 0.55), (1.5, 2.5), (0.08, 0.22), (0.45, 1.5), (0.4, 1.2), (0.35, 0.9), (0.25, 0.7), (0.25, 0.55), (0.4, 0.9), (0.5, 1.1)]
# All fields are semantic inputs, never checkpoint IDs or expert selectors.
FIELDS={'task_onehot':(0,12),'intent_onehot':(12,16),'task_value':16,'task_valid':17,'root_speed':18,'root_yaw_rate':19,'root_valid':(20,22),'goal_distance':22,'goal_sin':23,'goal_cos':24,'goal_valid':25,'reach_height':26,'reach_height_valid':27,'time_features':(28,45),'duration':45}
INTENTS=['free_motion','endpoint_motion','place_hold_retract','start_stop_departure'];NF=46

def features(tid,cmd,phase,intent=0,root=None,goal=None,height=None,frames=120,task_valid=True):
 device=phase.device;b,n=phase.shape;c=torch.zeros(b,n,NF,device=device);c[:,:,tid]=1;c[:,:,12+intent]=1
 if tid<11 and task_valid:
  lo,hi=RANGES[tid];c[:,:,16]=((cmd-lo)/(hi-lo)*2-1).clamp(-5,5)[:,None];c[:,:,17]=1
 if root is not None:c[:,:,18:22]=root
 if goal is not None:
  c[:,:,22]=goal[:,0,None]/3;c[:,:,23]=goal[:,1,None].sin();c[:,:,24]=goal[:,1,None].cos();c[:,:,25]=1
 if height is not None:c[:,:,26]=(height[:,None]-.84)/.03;c[:,:,27]=1
 freq=phase.new_tensor([1,2,3,4,6,8,10,12]);ph=phase[...,None]*freq*2*math.pi;c[:,:,28:45]=torch.cat([phase[...,None],ph.sin(),ph.cos()],-1);c[:,:,45]=frames/120
 return c

def availability(c):return c[...,17:18].maximum(c[...,20:22].amax(-1,keepdim=True)).maximum(c[...,25:26]).maximum(c[...,27:28])
