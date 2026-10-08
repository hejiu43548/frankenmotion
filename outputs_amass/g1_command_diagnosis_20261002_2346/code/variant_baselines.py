import os,json
from pathlib import Path
import numpy as np
import g1_runtime as rt
import original_experiment as ex
import run
OUT=Path(os.environ['DIAG_OUT'])
ex.OUT=OUT/'variant_baselines';ex.SOURCES=OUT/'human_variants'
m=rt.load_model();ex.retarget(m)
run.SOURCES=ex.OUT/'retarget';p=rt.Policy();rows=[]
for source in sorted(run.SOURCES.glob('*.npz')):
    r=run.evaluate(m,p,source.stem,1.,1.,'variant_'+source.stem)
    rows.append(r)
run.write('variant_baselines.json',rows)
