"""Auditable AMASS cache manifests and resumable, task-balanced sampling."""

import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

from .catalog import TASK_NAMES
from .model import file_sha256


class MotionDataset:
    def __init__(self, manifest, split, tasks, path_root=None, deduplicate=False):
        self.manifest = Path(manifest).resolve()
        root = Path(path_root).resolve() if path_root else self.manifest.parent
        records = json.loads(self.manifest.read_text())
        if not isinstance(records, list):
            raise ValueError("Dataset manifest must be a JSON list")
        self.rows = []
        for record in records:
            if record.get("split") != split:
                continue
            if record.get("task") not in tasks:
                continue
            if not record.get("family"):
                raise ValueError(
                    "Every training record needs a source family for split isolation"
                )
            if not record.get("cache"):
                raise ValueError(
                    "Staged supervision requires cache NPZ motion/local/local_mask/tx/quantity; prompt-only release manifests must first be prepared"
                )
            resolved = dict(record)
            path = Path(record["cache"])
            resolved["cache"] = str(path if path.is_absolute() else root / path)
            self.rows.append(resolved)
        missing = set(tasks) - {record["task"] for record in self.rows}
        if missing:
            raise ValueError(
                f"Missing {split} task data: {sorted(missing)}. Prepare all configured task caches first."
            )
        if deduplicate:
            unique = {}
            for record in self.rows:
                identity = (record["family"], record.get("key", record["cache"]))
                if identity not in unique or record.get("target_frames", 0) > unique[
                    identity
                ].get("target_frames", 0):
                    unique[identity] = record
            self.rows = list(unique.values())
        self.groups = {}
        self.items = []
        self.cache_hashes = {}
        for record_index, record in enumerate(self.rows):
            task_index = TASK_NAMES.index(record["task"])
            self.cache_hashes[record["cache"]] = file_sha256(record["cache"])
            with np.load(record["cache"], allow_pickle=False) as archive:
                item = {
                    name: torch.from_numpy(np.asarray(archive[name]).copy())
                    for name in ["motion", "local", "local_mask", "tx", "quantity"]
                }
            for name in ["motion", "local", "tx", "quantity"]:
                item[name] = item[name].float()
            frames = len(item["motion"])
            if (
                not 3 <= frames <= 120
                or item["motion"].shape != (frames, 205)
                or item["local"].shape != (frames, 408)
                or item["local_mask"].shape != (frames, 408)
                or item["tx"].shape != (512,)
            ):
                raise ValueError(f"Invalid cache shapes: {record['cache']}")
            if any(
                not torch.isfinite(item[name]).all()
                for name in ["motion", "local", "tx", "quantity"]
            ):
                raise ValueError(f"Nonfinite cache: {record['cache']}")
            item["task"] = torch.tensor(task_index)
            self.items.append(item)
            self.groups.setdefault(task_index, []).append(record_index)
        self.fingerprint = hashlib.sha256(
            json.dumps(self.cache_hashes, sort_keys=True).encode()
        ).hexdigest()
        missing = set(tasks) - {TASK_NAMES[index] for index in self.groups}
        if missing and not deduplicate:
            raise ValueError(
                f"Missing {split} task data: {', '.join(sorted(missing))}. No synthetic task duplication is allowed."
            )

    def batch(self, indices, device):
        items = [self.items[index] for index in indices]
        lengths = torch.tensor([len(item["motion"]) for item in items], device=device)
        result = {
            name: torch.stack(
                [
                    functional.pad(item[name], (0, 0, 0, 120 - len(item[name])))
                    for item in items
                ]
            ).to(device)
            for name in ["motion", "local", "local_mask"]
        }
        for name in ["tx", "quantity", "task"]:
            result[name] = torch.stack([item[name] for item in items]).to(device)
        result["lengths"] = lengths
        result["mask"] = torch.arange(120, device=device)[None] < lengths[:, None]
        return result


class StatefulSampler:
    def __init__(self, dataset, seed, root_stage=False):
        self.groups = dataset.groups
        self.count = len(dataset.rows)
        self.random = random.Random(seed)
        self.root_stage = root_stage
        self.pending = []
        self.epoch = 0
        self.seen = set()

    def batches(self, batch_size, accumulate=1):
        if self.root_stage:
            if not self.pending:
                self.pending = list(range(self.count))
                self.random.shuffle(self.pending)
            output = []
            for _ in range(accumulate):
                indices = self.pending[:batch_size]
                del self.pending[:batch_size]
                output.append(indices)
                self.seen.update(indices)
                if not self.pending:
                    self.epoch += 1
                    break
            return output
        indices = []
        for _ in range(batch_size):
            if not self.pending:
                self.pending = sorted(self.groups)
                self.random.shuffle(self.pending)
            task_index = self.pending.pop()
            indices.append(self.random.choice(self.groups[task_index]))
        self.seen.update(indices)
        return [indices]

    def state_dict(self):
        return dict(
            random=self.random.getstate(),
            pending=list(self.pending),
            epoch=self.epoch,
            seen=sorted(self.seen),
        )

    def load_state_dict(self, state):
        self.random.setstate(state["random"])
        self.pending = list(state["pending"])
        self.epoch = state["epoch"]
        self.seen = set(state["seen"])


def assert_disjoint(training, validation):
    overlap = {row["family"] for row in training.rows} & {
        row["family"] for row in validation.rows
    }
    if overlap:
        raise ValueError(
            f"Training/validation source-family leakage: {sorted(overlap)[:5]}"
        )
