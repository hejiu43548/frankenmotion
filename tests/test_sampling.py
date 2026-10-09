import unittest, collections, tempfile, json
from pathlib import Path
from shared_motion.data import BalancedTaskSampler, read_manifest


class SamplingTests(unittest.TestCase):
    def test_balance_without_truncation(self):
        rows = [
            {"task": t}
            for t, n in [("rare", 2), ("large", 1003), ("small", 13)]
            for _ in range(n)
        ]
        sampler = BalancedTaskSampler(rows, 9000, 123)
        draws = list(sampler)
        counts = collections.Counter(rows[i]["task"] for i in draws)
        self.assertEqual(set(counts.values()), {3000})
        self.assertEqual(len(sampler.groups["large"]), 1003)
        self.assertTrue(any(i > 100 for i in draws))
        self.assertEqual(draws, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(draws, list(sampler))

    def test_split_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.json"
            p.write_text(
                json.dumps(
                    [
                        {"task": "walk", "split": "train", "path": "x.npz"},
                        {"task": "walk", "split": "test", "path": "y.npz"},
                        {"task": "walk", "path": "z.npz"},
                    ]
                )
            )
            rows = read_manifest(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["path"], str(Path(d) / "x.npz"))

    def test_partial_cycle(self):
        rows = [{"task": str(i)} for i in range(20)]
        draws = list(BalancedTaskSampler(rows, 43))
        counts = collections.Counter(draws)
        self.assertEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
