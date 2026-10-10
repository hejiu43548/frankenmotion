#!/usr/bin/env bash
set -euo pipefail
run_mode="${1:-physics}"
run_name="${2:-review_$(date -u +%Y%m%d_%H%M%S)}"
[[ "$run_mode" == physics || "$run_mode" == full ]]
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]
ssh betail bash -s -- "$run_mode" "$run_name" <<'REMOTE'
set -euo pipefail
run_mode="$1"
run_name="$2"
demo_root=/mnt/sda2/frankenmotion/outputs_amass/g1_demo_3_20261010
demo_python=/home/psirobot/projects/frankenmotion/.venv_unified/bin/python
demo_code="$demo_root/attempts/symmetric_first/code"
demo_entry="$demo_code/scripts/g1_demo3.py"
export PYTHONPATH=/home/psirobot/.cache/g1_demo_20261010_deps:/tmp/frankenmotion_hydra_deps
export MUJOCO_GL=egl
run_root="$demo_root"
if [[ "$run_mode" == full ]]; then
    run_root="$demo_root/reproductions/$run_name"
    [[ ! -e "$run_root" ]]
    PYTHONPATH=/tmp/frankenmotion_hydra_deps "$demo_python" "$demo_entry" phase=generate "output=$run_root" generation_device=cpu
    "$demo_python" "$demo_entry" phase=retarget "output=$run_root"
fi
"$demo_python" "$demo_entry" phase=compose "output=$run_root" "attempt=$run_name"
"$demo_python" "$demo_entry" phase=verify "output=$run_root" "attempt=$run_name"
"$demo_python" "$demo_entry" phase=render "output=$run_root" "attempt=$run_name"
"$demo_python" "$demo_code/scripts/g1_demo3_evidence.py" "$run_root/attempts/$run_name"
"$demo_python" - "$demo_root" "$run_root" "$run_name" "$demo_code/scripts" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[4])
from g1_demo3_verify import compare

original_root, repeated_root, repeated_attempt = sys.argv[1:4]
comparison = compare(original_root, repeated_root, "symmetric_first", repeated_attempt)
folder = Path(repeated_root) / "attempts" / repeated_attempt
(folder / "reproduction_comparison.json").write_text(json.dumps(comparison, indent=2))
acceptance = json.loads((folder / "acceptance.json").read_text())
assert acceptance["passed"], acceptance
print(json.dumps(dict(reproduction_arrays_equal=True, acceptance=acceptance)))
PY
printf 'Result: %s/attempts/%s\n' "$run_root" "$run_name"
REMOTE
