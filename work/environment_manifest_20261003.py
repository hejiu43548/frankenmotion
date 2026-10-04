import sys,json,platform,importlib.metadata as md
from pathlib import Path
packages=['torch','numpy','scipy','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','mjlab','onnxruntime','mink','qpsolvers','daqp','transformers','hydra-core','omegaconf','matplotlib']
record=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={})
for name in packages:
 try:record['packages'][name]=md.version(name)
 except md.PackageNotFoundError:pass
print(json.dumps(record,indent=2))
