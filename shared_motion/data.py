"""Task-uniform sampling without truncating or duplicating the stored corpus."""

import json, random
from pathlib import Path
from collections import defaultdict


def read_manifest(path, split="train"):
    path = Path(path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list of records")
    selected = []
    for original in rows:
        if original.get("split") != split:
            continue
        row = dict(original)
        if not row.get("task"):
            raise ValueError("Every sample needs a task label")
        for field in ["path", "prompt_cache"]:
            if row.get(field):
                p = Path(row[field])
                row[field] = str(p if p.is_absolute() else (path.parent / p).resolve())
        selected.append(row)
    if not selected:
        raise ValueError(f"No samples explicitly marked split={split!r}")
    return selected


class BalancedTaskSampler:
    """Shuffle a full task cycle; then sample uniformly within each task.

    An epoch contains `num_samples` draws, with replacement within tasks. Each
    task's draw count differs by at most one. Corpus size is never capped.
    `set_epoch` makes resumed/distributed callers able to choose a new seed.
    """

    def __init__(self, rows, num_samples, seed=0):
        if not rows or num_samples <= 0:
            raise ValueError("Nonempty rows and positive num_samples required")
        self.groups = defaultdict(list)
        for i, row in enumerate(rows):
            self.groups[row["task"]].append(i)
        self.tasks = sorted(self.groups)
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        remaining = self.num_samples
        while remaining:
            tasks = self.tasks.copy()
            rng.shuffle(tasks)
            for task in tasks:
                if not remaining:
                    return
                yield rng.choice(self.groups[task])
                remaining -= 1
