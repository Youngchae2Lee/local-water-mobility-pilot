# Local Water Structure → Mobility: Mini Pilot

Repository: https://github.com/Youngchae2Lee/local-water-mobility-pilot

Can tetrahedral order q add predictive information beyond local density and coordination for future oxygen mobility at 298 K?

**Status: synthetic software demonstration only. Real-water training is on hold.** This is a structure-to-mobility regression pilot, not an interatomic potential. No water dataset, trained water model, or water performance claim is included.

## Run

Use Python with NumPy. The optional four-model smoke test also requires scikit-learn. See `requirements.txt`; CPU is sufficient, and no GPU is required.

```bash
python -B -m unittest -v test_demo
python -B demo.py
python -B demo.py --with-models
```

The demo generates artificial tabular features and labels entirely in memory. Its density, coordination and q columns are placeholders, not values computed from molecular coordinates. An artificial q contribution exercises the comparison; any resulting improvement is built into the generator and is not scientific evidence. The CLI accepts no external dataset path.

All molecules at an origin share a time split. Target intervals crossing split boundaries are excluded. Scaling and fitting use training rows only. The four models are a training-mean predictor, two Ridge regressions with/without q, and one small Random Forest. Scores retain negative R². `paired_q_gain` is baseline minus augmented error (positive is better); the planned scientific report instead uses augmented minus baseline MAE (negative is better).

## Planned water protocol

The proposed target is the same oxygen's 10 ps squared TOR displacement in Å², reconstructed at every 1 ps frame using an oxygen reference, the [0,L) cell and next-frame box; no COM drift subtraction. Features are evaluated at the origin from periodic structural coordinates: density = 5/(4πd5³/3), coordination at O–O distance ≤3.5 Å, and four-neighbor tetrahedral q. Fourth/fifth neighbor ties exclude that whole origin frame.

The planned half-open frame ranges are train [0,600), validation [650,800), test [850,1000), with a 50 ps gap and 10 ps origin stride. These are a design for one designated dataset, not defaults transferable to arbitrary trajectories. TOR does not remove every barostat effect. A gap does not certify independence, equilibrium or convergence.

Models: Ridge alpha=1; RF 50 trees, max_depth=4, min_samples_leaf=10, n_jobs=1; seed=20261014. No large tuning or deep learning. Validation checks execution; no train+validation refit. Test is reserved for one final evaluation after the design is locked. Report MAE/RMSE/R², paired Ridge ΔMAE and 50 ps block errors on a common cohort. Mixed or null results remain valid outcomes. A single-run result is not cross-system generalization, causality or isoconfigurational propensity.

| Model | Water MAE | Water RMSE | Water R² |
|---|---|---|---|
| Training mean | — | — | — |
| Density + coordination Ridge | — | — | — |
| Density + coordination + q Ridge | — | — | — |
| Small Random Forest | — | — | — |

No water training is scheduled by this public demo. Data-use authorization and the separate research hold must be resolved before any such work. The public release does not authorize publication of laboratory data or results.

## Release boundary

Only this directory is the publication candidate. Raw trajectories, topologies, molecular rows, laboratory-derived summaries, server details, private notes, remote runners, and research-vendored implementations are excluded. This release therefore does not implement the planned molecular geometry or NPT adapter. See `PUBLICATION_POLICY.md` and `VALIDATION.md`.

This is a personal learning project, not a claim of institutional approval. A software license has not yet been selected; no license for excluded research code or data is implied.
