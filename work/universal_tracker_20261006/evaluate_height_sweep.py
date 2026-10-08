import subprocess
from pathlib import Path
R=Path("/home/pku/frankenmotion");D=R/"outputs_amass/universal_tracker_20261006";W=R/"work/universal_tracker_20261006"
for name,ck in [("height_stable","backup/stable_frozen/policy.pt"),("height_v3_3000","training/joint_v3_broad_init/model_3000.pt"),("height_v4_1000","training/joint_v4_long_physical/model_1000.pt")]:
 with (D/(name+".log")).open("w") as f:r=subprocess.run([str(R/"work/mjlab_stable_env/bin/python"),str(W/"evaluate_table_alternative.py"),"--checkpoint",str(D/ck),"--name",name,"--scenes-folder",str(D/"reach_height_scenes")],stdout=f,stderr=subprocess.STDOUT)
 print(name,r.returncode,flush=True)
 if r.returncode:break
