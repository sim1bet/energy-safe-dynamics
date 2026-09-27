# Release file map and audit

| Dataset | Frozen champion / status | Checkpoint | Inputs | Final figure exports |
| --- | --- | --- | --- | --- |
| `duffing_doublewell` | `duffing_v5_champion_exact_seed4`, seed 4 | `checkpoints/params.pkl` | `parameters/duffing_v1.npz` | Hamiltonian and epsilon-sweep PNGs |
| `deep_dissipative_nlink_n3` | `copilot_A1_nlink3_stageA_recipe`, seed 0 | `checkpoints/params.pkl` | verbatim n=3 upstream arrays | Hamiltonian 2D/3D and epsilon-sweep PNGs |
| `deep_dissipative_nlink_n2` | `copilot_A1_full_stageA_recipe`, seed 0 | `checkpoints/params.pkl` | verbatim n=2 upstream arrays | Hamiltonian 2D/3D and epsilon-sweep PNGs |
| `nanodrone_S3` | S3/T2 seed 47, selected physical-MAE model | `checkpoints/best_physical_mae.pkl` | exact benchmark CSVs | publication Hamiltonian 2D/3D and epsilon PNGs |
| `CED` | supplied champion configuration | `checkpoints/params.pkl` | `DATAUNIF.MAT` | generated on demand |
| `Silverbox` | supplied pHEBM configuration | initialized parameters (no trained pHEBM checkpoint supplied) | CSV and MAT benchmark files | generated on demand |

The unrelated n-link n=5 run is excluded. No raw training logs,
TensorBoard/W&B outputs, logs, IDE files, caches, notebooks, or development-only
reports are included.

`Stress_test_final/Stress_tests` is the minimal Python support package required
by the copied stress-test entry points. It contains source code only; generated
test outputs are not included. Every retained champion configuration and its
checkpoint reside under the same dataset directory.

Source archives and duplicate mirrors were omitted when their extracted bytes
were already present. The top-level `baselines/` mirror from the primary input
was likewise removed; comparator assets have one canonical location below each
dataset.

## Comparator assets

- n-link n=2 and n=3: upstream DDM `naive` and `dissipative` configurations and
  final `best.checkpoint` files are in `parameters/baselines/`; runnable DDM
  source is in `dataset_interfaces/baseline_interfaces/deep_dissipative_model/`.
- PortHNN-u: the canonical implementation is `porthnn_u/`; frozen Duffing,
  three-link, Silverbox, and CED champions are in `baselines/porthnn_u/`.
  Former dataset-interface module locations are compatibility wrappers only.
- Nanodrone: S3 BB seed-47 configuration, final checkpoint, model
  implementation, and loader are under `champion_configuration/baselines/`,
  `checkpoints/baselines/`, and `dataset_interfaces/baseline_interfaces/nanodrone_bb/`.

## Robustness artifacts

`certificates/hner99/` contains nine supplied frozen runtime HNER99 reports,
their text renderings, an aggregate summary, and an explicit checkpoint
provenance registry. `scripts/compute_robustness_radii.py` recomputes supported
models without overwriting the frozen reports. The missing historical
Silverbox pHEBM checkpoint and NanoDrone seed-23/seed-47 distinction are
recorded rather than silently substituted.

`runtime_hner99/` is the executable library: it contains the complete numerical
protocol, release-relative model registry, fixed-budget audit workflow, and a
machine-readable index pointing to every canonical frozen report. The scripts
folder contains thin reviewer-facing launchers for that package.
