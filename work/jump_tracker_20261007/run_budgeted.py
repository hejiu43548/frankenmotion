"""Bounded subprocess group runner; only terminates its own child group."""
import argparse,subprocess,os,signal,time,json,datetime
from pathlib import Path
D=Path('/home/pku/frankenmotion/outputs_amass/jump_tracker_20261007');D.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--seconds',type=float,default=10000);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();assert a.command
hard=datetime.datetime(2026,10,6,20,10,tzinfo=datetime.timezone.utc).timestamp();seconds=min(a.seconds,hard-time.time());assert seconds>0
status=D/(a.label+'_process.json');log=(D/(a.label+'.log')).open('w');child=subprocess.Popen(a.command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/home/pku/frankenmotion',env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));r=dict(label=a.label,pid=child.pid,command=a.command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_s=seconds,hard_compute_cutoff_utc='2026-10-06T20:10:00Z');status.write_text(json.dumps(r,indent=2))
try:rc=child.wait(timeout=seconds)
except subprocess.TimeoutExpired:
 os.killpg(child.pid,signal.SIGTERM)
 try:child.wait(timeout=5)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 rc=124
r.update(returncode=rc,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());status.write_text(json.dumps(r,indent=2));log.close();raise SystemExit(rc)
