import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.train_strike3d import load_items
from shared_motion.training.model import file_sha256


class PortableStrikeDataTests(unittest.TestCase):
    def test_relocated_cache_requires_exact_hash_and_casts_floats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir()
            cache = root / "cache/sample.npz"
            np.savez(
                cache,
                motion=np.zeros((3, 205)),
                local=np.zeros((3, 408)),
                local_mask=np.zeros((3, 408), dtype=bool),
                tx=np.zeros(512),
                hands=np.int64(1),
                positions=np.zeros(3),
                event_frames=np.int64(1),
            )
            rows = [
                dict(
                    cache="/nonexistent-original-host/sample.npz",
                    rebuilt_cache_sha256=file_sha256(cache),
                )
            ]
            (root / "train.json").write_text(json.dumps(rows))
            _, items = load_items(root, "train")
            self.assertEqual(items[0]["motion"].dtype, torch.float32)
            self.assertEqual(items[0]["local_mask"].dtype, torch.bool)
            self.assertEqual(items[0]["hands"].dtype, torch.long)
            with cache.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                load_items(root, "train")


if __name__ == "__main__":
    unittest.main()
