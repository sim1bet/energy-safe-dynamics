# Validation report

## Completed in this build environment

- Recursive inventories and relative-path/hash comparisons completed.
- All JSON files parse successfully.
- All Python sources compile successfully with `compileall`.
- `python validate_release.py --structural` passes for all six datasets and
  invokes the dependency-light PortHNN-u and HNER99 artifact audits.
- All four frozen PortHNN-u checkpoint hashes match their manifests; every NPZ
  array is nonempty and finite. Stored Duffing and three-link metrics reproduce,
  and the Silverbox/CED validation-selection records are internally consistent.
- All nine supplied HNER99 JSON reports pass required-field, status, shell,
  numerical-radius, and projection-error checks. Recorded checkpoint hashes
  match for CED pHEBM, CED PortHNN-u, and Silverbox PortHNN-u.
- The dependency-light `runtime_hner99` package and artifact API import
  successfully; all nine JSON/text report pairs resolve through its canonical
  artifact index.
- The privacy/portability scan reports no user path, cluster path, scheduler
  directive, cache directory, or compiled Python artifact.

## Runtime validation status

`scripts/validate_inference.py` is the reproducible runtime test. It loads each
dataset, applies its champion configuration, loads the retained checkpoint when
available, and requires finite states, port outputs, and observations from a
four-step rollout.

The current build container does not provide JAX and its package index exposes
no JAX wheel, so the runtime suite could not be executed here. This is an
environment limitation, not a recorded pass. Run the following in a fresh
environment after installing `requirements.txt`:

```bash
python validate_release.py
python scripts/validate_porthnn_u_runtime.py
```

Any import, dataset, checkpoint, shape, or non-finite failure makes the command
exit nonzero and is reported per dataset; there is no broad silent fallback.
