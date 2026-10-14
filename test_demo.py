import unittest
import numpy as np
import demo


class DemoTests(unittest.TestCase):
    def test_split_boundary_and_common_cohort(self):
        rows = demo.synthetic_rows()
        parts, excluded = demo.split_rows(rows, demo.RANGES, 12, 3)
        self.assertEqual([len(parts[k]) for k in demo.RANGES], [1692, 396, 396])
        self.assertEqual(len(excluded), 360)
        owners = {}
        for name, part in parts.items():
            lo, hi = demo.RANGES[name]
            for r in part:
                self.assertTrue(lo <= r['frame_id'] < r['target_end_frame'] < hi)
                owners.setdefault(r['frame_id'], set()).add(name)
            for frame in {r['frame_id'] for r in part}:
                self.assertEqual(sum(r['frame_id'] == frame for r in part), 12)
        self.assertTrue(all(len(v) == 1 for v in owners.values()))

    def test_inadequate_gap_rejected(self):
        with self.assertRaises(ValueError):
            demo.split_rows([], demo.RANGES, 2, 3)

    def test_negative_and_undefined_r2(self):
        self.assertLess(demo.metrics([1, 2, 3], [10, 10, 10])['r2'], 0)
        self.assertIsNone(demo.metrics([1, 1], [2, 2])['r2'])

    def test_models_train_only_same_holdout(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest('scikit-learn unavailable; model test not run')
        parts, _ = demo.split_rows(demo.synthetic_rows(), demo.RANGES, 12, 3)
        models = demo.fit_train_only(parts['train'], 20261014)
        scaler = models['ridge_density_coord_q'][0].named_steps['standardscaler']
        x = np.array([[r['density'], r['coordination'], r['q']] for r in parts['train']])
        np.testing.assert_allclose(scaler.mean_, x.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(x))
        result = demo.evaluate(models, parts['test'])
        self.assertEqual({result[m]['overall']['n'] for m in demo.MODELS}, {396})


if __name__ == '__main__':
    unittest.main()
