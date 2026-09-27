# PortHNN-u and HNER99 integration manifest

Created before modifying the consolidated release.

| Area | Existing release | New source | Resolution | Validation |
|---|---|---|---|---|
| Duffing PortHNN-u | Two identical `porthnn_u_jax.py` copies | Primary adversarial `porthnn_u/duffing.py` (byte-identical implementation) and frozen seed-0 champion | One canonical top-level module; old import locations become wrappers; champion and audit artifacts retained once | Checkpoint hash/schema, parameter count, finite model smoke test when JAX is available |
| Three-link PortHNN-u | Older full-matrix implementation and no complete champion bundle | Primary adversarial corrected `three_link.py`, configuration, checkpoint, predictions, normalization, metrics | Corrected primary implementation authoritative; existing legacy module becomes wrapper | Checkpoint leaf/schema checks, stored-prediction/metric audit, finite rollout when JAX is available |
| Silverbox/CED PortHNN-u | No integrated champions | Secondary adversarial modular implementation and best-validation champions | Integrate modular modules into the same `porthnn_u` namespace; retain frozen configs, normalization, predictions, diagnostics, and parameters | SHA-256, parameter reconstruction, J/R/G/storage/readout smoke tests, validation-prediction reproduction |
| CLI | Existing standalone scripts; primary and secondary CLIs conflict | Both CLIs | Unified subcommand parser exposing Duffing, three-link, Silverbox, and CED workflows | `--help`/parser smoke tests and command registry audit |
| HNER99 | No aggregate release command | Runtime HNER99 library, algorithm, audit workflow, and hard-coded historical loaders | Integrate an importable `runtime_hner99` package; preserve the numerical algorithm; replace loaders with release-relative adapters and explicit selector/artifact registries | Package import; artifact-index audit; optional recomputation per selector; fixed 32/64-shell audit |
| Generated results | Existing dataset figures/certificates | HNER99 JSON/TXT reports | Canonical `certificates/hner99/` tree with normalized checkpoint paths; no duplicate copies | JSON schema, checkpoint existence, selector/model coverage |
| Cluster launchers | None in public release | Four scheduler-specific scripts with private paths | Excluded from active release; portable Python commands documented | Privacy/HPC scan |

No scientific result will be silently recomputed or overwritten during integration. Frozen reports remain labeled as supplied results; fresh computations write to an explicit output directory.
