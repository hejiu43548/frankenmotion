#!/usr/bin/env bash
set -eu
run="${1:-/home/pku/frankenmotion/outputs_amass/root_control_20260925}"
cat "$run/status.json"
echo
if [[ -f "$run/exit.json" ]]; then cat "$run/exit.json"; echo; fi
if [[ -f "$run/DONE" ]]; then
  echo 'SUCCESS: training and final evaluation completed; best.pt is available.'
elif [[ -f "$run/train.pid" ]] && kill -0 "$(cat "$run/train.pid")" 2>/dev/null; then
  echo 'RUNNING: process is alive.'
else
  echo 'STOPPED WITHOUT DONE: not successful completion; inspect train.log.'
fi
