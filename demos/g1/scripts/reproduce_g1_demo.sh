#!/usr/bin/env bash
# Replay physics from preserved references, or rebuild all four layers in a new folder.
set -euo pipefail
run_mode="${1:-physics}"
run_name="${2:-review_$(date -u +%Y%m%d_%H%M%S)}"
[[ "$run_mode" == physics || "$run_mode" == full ]]
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]
ssh betail bash -s -- "$run_mode" "$run_name" <<'REMOTE'
set -euo pipefail
run_mode="$1"
run_name="$2"
demo_root=/mnt/sda2/frankenmotion/outputs_amass/g1_demo_1_20261010
demo_python=/home/psirobot/projects/frankenmotion/.venv_unified/bin/python
demo_entry="$demo_root/attempts/demo1_final/code/scripts/g1_demo.py"
physics_dependencies=/home/psirobot/.cache/g1_demo_20261010_deps:/tmp/frankenmotion_hydra_deps
run_root="$demo_root"
if [[ "$run_mode" == full ]]; then
    run_root="$demo_root/reproductions/$run_name"
    [[ ! -e "$run_root" ]]
    PYTHONPATH=/tmp/frankenmotion_hydra_deps "$demo_python" "$demo_entry" phase=generate "output=$run_root" generation_device=cpu
    PYTHONPATH="$physics_dependencies" "$demo_python" "$demo_entry" phase=retarget "output=$run_root"
fi
PYTHONPATH="$physics_dependencies" "$demo_python" "$demo_entry" phase=compose "output=$run_root" "attempt=$run_name"
PYTHONPATH="$physics_dependencies" "$demo_python" "$demo_entry" phase=audit "output=$run_root" "attempt=$run_name"
PYTHONPATH="$physics_dependencies" "$demo_python" "$demo_entry" phase=render "output=$run_root" "+render_dir=attempts/$run_name" +video=true
printf 'Result: %s/attempts/%s\n' "$run_root" "$run_name"
REMOTE
