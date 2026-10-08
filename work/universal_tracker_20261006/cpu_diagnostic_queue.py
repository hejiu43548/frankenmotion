import json,subprocess
from pathlib import Path
R=Path("/home/pku/frankenmotion");D=R/"outputs_amass/universal_tracker_20261006";W=R/"work/universal_tracker_20261006"
spec=json.loads((D/"cpu_diagnostic_queue.json").read_text())
for item in spec:
 cmd=[str(R/"work/mjlab_stable_env/bin/python"),str(W/"evaluate_cpu_general.py")]
 for k,v in item.items():cmd.extend(["--"+k,str(v)])
 with (D/(item["name"]+".log")).open("w") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(item["name"],r.returncode,flush=True)
 if r.returncode:break
 if item["name"].startswith("eleven_"):subprocess.run([str(R/"work/mjlab_stable_env/bin/python"),str(W/"assess_native_eleven.py"),"--name",item["name"]],check=True)
