import subprocess
from pathlib import Path
R=Path("/home/pku/frankenmotion");D=R/"outputs_amass/universal_tracker_20261006"
for alpha in [1.,.8,.6,.4]:
 name="v3_3000_filter_"+str(alpha).replace(".","p")
 with (D/(name+".log")).open("w") as f:
  r=subprocess.run([str(R/"work/mjlab_stable_env/bin/python"),str(R/"work/universal_tracker_20261006/evaluate_filtered_table.py"),"--checkpoint",str(D/"training/joint_v3_broad_init/model_3000.pt"),"--name",name,"--filter-alpha",str(alpha)],stdout=f,stderr=subprocess.STDOUT)
 print(name,r.returncode,flush=True)
 if r.returncode:break
