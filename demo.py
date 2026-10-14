"""Public synthetic tabular QA; no trajectory reader or real-water training."""
import argparse
import json
import platform
import numpy as np

MODELS = ('training_mean', 'ridge_density_coord', 'ridge_density_coord_q', 'small_rf')
RANGES = {'train': (0, 144), 'validation': (156, 192), 'test': (204, 240)}


def split_rows(rows, ranges, gap_frames, lag_frames):
    if gap_frames < lag_frames or gap_frames < 0:
        raise ValueError('Gap must cover lag and prespecified correlation horizon')
    if set(ranges) != {'train', 'validation', 'test'}:
        raise ValueError('Three named splits required')
    ordered = [ranges[k] for k in ('train', 'validation', 'test')]
    if any(lo < 0 or hi <= lo for lo, hi in ordered):
        raise ValueError('Invalid split interval')
    if any(right[0] - left[1] < gap_frames for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Overlapping splits or inadequate gap')
    splits = {k: [] for k in ranges}
    excluded = []
    for row in rows:
        f, end = row['frame_id'], row['target_end_frame']
        if end - f != lag_frames:
            raise ValueError('Inconsistent target interval')
        dest = next((name for name, (lo, hi) in ranges.items() if lo <= f and end < hi), None)
        (splits[dest] if dest else excluded).append(row)
    if any(not v for v in splits.values()):
        raise ValueError('Empty split')
    return splits, excluded


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    error = pred - y
    denom = np.sum((y - y.mean()) ** 2)
    return {'mae': float(np.mean(abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'r2': float(1 - np.sum(error ** 2) / denom) if denom > 0 else None, 'n': len(y)}


def fit_train_only(train, seed):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.array([[r['density'], r['coordination'], r['q']] for r in train])
    y = np.array([r['y'] for r in train])
    models = {
        'training_mean': (DummyRegressor(strategy='mean'), [0, 1, 2]),
        'ridge_density_coord': (make_pipeline(StandardScaler(), Ridge(alpha=1.)), [0, 1]),
        'ridge_density_coord_q': (make_pipeline(StandardScaler(), Ridge(alpha=1.)), [0, 1, 2]),
        'small_rf': (RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10,
                                          random_state=seed, n_jobs=1), [0, 1, 2])}
    for model, columns in models.values():
        model.fit(x[:, columns], y)
    return models


def evaluate(models, rows, block_frames=20):
    x = np.array([[r['density'], r['coordination'], r['q']] for r in rows])
    y = np.array([r['y'] for r in rows])
    frames = np.array([r['frame_id'] for r in rows])
    block_ids = (frames - frames.min()) // block_frames
    result = {}
    for name, (model, cols) in models.items():
        pred = model.predict(x[:, cols])
        result[name] = {'overall': metrics(y, pred), 'blocks': {
            str(b): metrics(y[block_ids == b], pred[block_ids == b]) for b in np.unique(block_ids)}}
    result['paired_q_gain'] = {key: result['ridge_density_coord']['overall'][key] -
        result['ridge_density_coord_q']['overall'][key] for key in ('mae', 'rmse')}
    result['paired_q_gain']['r2'] = (result['ridge_density_coord_q']['overall']['r2'] -
                                    result['ridge_density_coord']['overall']['r2'])
    return result

def synthetic_rows(seed=20261014):
    """Artificial predictors and target, not molecular geometry or dynamics."""
    rng = np.random.default_rng(seed)
    rows = []
    for frame in range(237):
        for molecule in range(1, 13):
            density = rng.uniform(.02, .05)
            coordination = int(rng.integers(2, 8))
            q = rng.uniform(-.5, 1.)
            y = 5 + 20 * density + .2 * coordination + .8 * q + rng.uniform(-.2, .2)
            rows.append(dict(frame_id=frame, target_end_frame=frame+3,
                             molecule_id=molecule, density=density,
                             coordination=coordination, q=q, y=y))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-models', action='store_true')
    args = parser.parse_args()
    parts, excluded = split_rows(synthetic_rows(), RANGES, 12, 3)
    result = dict(scope='SYNTHETIC TABULAR QA ONLY; q signal deliberately injected',
                  seed=20261014, rows_by_split={k:len(v) for k,v in parts.items()},
                  purged_rows=len(excluded), python=platform.python_version(), numpy=np.__version__)
    if args.with_models:
        import sklearn
        models=fit_train_only(parts['train'], 20261014)
        result['sklearn']=sklearn.__version__
        result['synthetic_validation']=evaluate(models, parts['validation'])
        result['synthetic_test']=evaluate(models, parts['test'])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
