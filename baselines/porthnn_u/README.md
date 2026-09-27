# PortHNN-u adversarial baselines

This directory contains the frozen PortHNN-u champions used in the adversarial
comparison. The executable implementation is the single top-level
`porthnn_u/` package.

| Dataset | Configuration | Frozen checkpoint |
|---|---|---|
| Duffing double well | `configs/duffing/seed0.yaml` | `champions/duffing/seed0/parameters.npz` |
| Three-link pendulum | `configs/three_link/full_jrg.yaml` | `champions/three_link/full_jrg/seed0/checkpoint_best.npz` |
| Silverbox | `configs/silverbox/sb02_16.yaml` | `champions/silverbox/sb02_16/best_validation_parameters.npz` |
| CED | `configs/ced/ced02_07.yaml`, retained fold 1 | `champions/ced/ced02_07/fold_1/best_validation_parameters.npz` |

Run the dependency-light artifact audit with:

```bash
python scripts/validate_porthnn_u_artifacts.py
```

After installing the full environment, execute all four models and check the
port-Hamiltonian structure with:

```bash
python scripts/validate_porthnn_u_runtime.py
```

Training remains available through `python -m porthnn_u.cli`. The bundled
checkpoints are the selected best-validation archives; validation data, test
predictions, metrics, normalization, and provenance are retained beside them.
