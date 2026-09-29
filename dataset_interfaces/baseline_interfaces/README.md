# Adversarial/baseline interfaces

`deep_dissipative_model/` is the retained upstream DDM source used by the
naive and dissipative n-link baselines. Install it locally with
`pip install ./dataset_interfaces/baseline_interfaces/deep_dissipative_model`.
The frozen n=2/n=3 configurations and final `best.checkpoint` files are in the
corresponding dataset's `parameters/baselines/{naive,dissipative}` directory.
Point the upstream scripts at the co-located `*nlink.*.npy` inputs.

`porthnn_u/` now contains compatibility imports only. The canonical JAX
implementation for Duffing, three-link, Silverbox, and CED is the top-level
`porthnn_u/` package. Frozen configurations, checkpoints, and audit artifacts
are under `baselines/porthnn_u/`; aggregate validators are in `scripts/`.

`nanodrone_bb/` contains the exact `ResidualQuadModel` definition and a loader
for the S3 seed-47 BB checkpoint:

```bash
python dataset_interfaces/baseline_interfaces/nanodrone_bb/load_checkpoint.py --smoke-rollout
```
