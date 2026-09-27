# Internal merge manifest (pre-integration)

This manifest was created before modifying the copied implementation.

| Area | Primary source | Secondary source | Resolution | Planned validation |
|---|---|---|---|---|
| EBM core | `EBM_model/*.py` | root `EBM_*.py` | Primary authoritative. Add only compatible mixed-polynomial activation and rollout substeps. Do not activate the secondary 39-field tank-specific parameter schema because supplied champions do not use it. | Import, initialized forward rollouts, all retained checkpoints. |
| CED/Silverbox interfaces | none | `CED_main.py`, `Silverbox_main.py`, `dataset_paths.py` | Integrate under `dataset_interfaces/`; adapt imports and release-relative paths. | Dataset load plus finite four-step inference. |
| CED checkpoint | none | `checkpoints/CED_best_params.pkl` | Preserve bytes; migrate legacy 20-field trunk into the primary current 31-field schema via named-field compatibility loader. | Load and execute checkpoint. |
| Stress tests | `Stress_test_final/Stress_tests` | `Stress_tests` | Primary files remain authoritative. Secondary-only generic adapter/experiment/tests are reviewed individually; private generated reports are excluded. | Imports and pytest collection/smoke. |
| Dependencies | primary `requirements.txt` | secondary `requirements.txt` and vendored wheels | Merge only direct runtime/test dependencies; exclude vendored site-packages. | Fresh-install metadata and import validation. |
| Logs/results | benchmark figures and curated primary assets | raw logs, rankings, generated validation outputs | Exclude raw run logs and stale generated reports; retain curated benchmark artifacts and checkpoint. | Privacy/path scan. |
| Paths/compute environment | mixed | mixed | Replace release-breaking assumptions with paths derived from `__file__`; exclude submission scripts and machine paths from active docs/artifacts. | Run validators outside repository cwd. |

Ambiguous scientific alternatives remain documented in `MERGE_REPORT.md`; none will be enabled silently.
