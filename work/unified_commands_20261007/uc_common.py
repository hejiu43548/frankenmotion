from pathlib import Path
import sys,json,math
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';B=D/'backup'
sys.path[:0]=[str(R/'work/unified_generator_20261007'),str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import common as prev
from core import torch,np,nn,FK,sample,TASKS,RANGES,HH,encode_control
from command_schema import FIELDS,INTENTS,NF,features,availability
