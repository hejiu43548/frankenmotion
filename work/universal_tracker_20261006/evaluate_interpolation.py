import subprocess
from pathlib import Path
R=Path("/home/pku/frankenmotion");D=R/"outputs_amass/universal_tracker_20261006";W=R/"work/universal_tracker_20261006";py=str(R/"work/mjlab_stable_env/bin/python")
for fraction in [.25,.5,.75]:
 name="fixed_interp_"+str(fraction).replace(".","p");ck=str(D/"fixed_weight_interpolation"/f"candidate_{fraction:.2f}.pt")
 commands=[[py,str(W/"evaluate_table.py"),"--checkpoint",ck,"--name",name],[py,str(W/"evaluate_cpu_general.py"),"--checkpoint",ck,"--name",name,"--manifest",str(D/"native_eleven_manifest.json")],[py,str(W/"assess_native_eleven.py"),"--name",name]]
 for i,cmd in enumerate(commands):
  with (D/(name+f"_{i}.log")).open("w") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  print(name,i,r.returncode,flush=True)
  if r.returncode:raise RuntimeError("Failed interpolation diagnostic")
