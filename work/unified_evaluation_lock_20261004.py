"""Serialize GPU rollout workers while concurrent large training runs use the GPU."""
from contextlib import contextmanager
from pathlib import Path
import fcntl

@contextmanager
def evaluation_slot():
    path=Path('/home/pku/frankenmotion/outputs_amass/franken_unified_20261004/native_evaluation.lock')
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)
