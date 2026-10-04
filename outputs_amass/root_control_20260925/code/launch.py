"""Detached supervisor records abnormal exits as well as successful completion."""
import json, os, subprocess, sys, time
from pathlib import Path
root=Path('/home/pku/frankenmotion')
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
script=Path(__file__).resolve().parent/'train.py'
command=[str(root/'.conda/bin/python'),'-u',str(script),'--out',str(out)]+sys.argv[2:]
(out/'supervisor.pid').write_text(str(os.getpid()))
with (out/'train.log').open('a') as log:
    process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    (out/'train.pid').write_text(str(process.pid))
    result=process.wait()
record={'exit_code':result,'finished':time.strftime('%Y-%m-%d %H:%M:%S'),'command':command}
(out/'exit.json').write_text(json.dumps(record,indent=2))
if result!=0:
    path=out/'status.json'; value=json.loads(path.read_text()) if path.exists() else {}
    value.update(state='failed',exit_code=result)
    path.write_text(json.dumps(value,indent=2))
sys.exit(result)
