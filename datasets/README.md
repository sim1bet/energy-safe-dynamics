# Datasets and benchmark artifacts

This directory contains the data-facing artifacts used by **Safe-by-design learning via energy-based neural network**. It is deliberately split between redistributable material and externally acquired benchmark data.

| Dataset | Included in this folder | Raw-data status | Canonical source |
|---|---|---|---|
| Duffing double well | Generated samples, configuration, checkpoint, and results | Project-generated; included | This repository |
| Deep dissipative n-link (2-link and 3-link) | Generated arrays, configurations, checkpoints, and results | Included under the upstream generator's MIT terms | [DeepDissipativeModel](https://github.com/kojima-r/DeepDissipativeModel) |
| Coupled Electric Drives (CED) | Configuration and checkpoint | Raw benchmark file excluded; download required | [Nonlinear System Identification Benchmarks](https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives) |
| Silverbox | Configuration | Raw benchmark files excluded; download required | [Nonlinear System Identification Benchmarks](https://www.nonlinearbenchmark.org/benchmarks/silverbox) |
| NanoDrone S3 | Configurations, checkpoints, and result figures | Raw flight logs excluded; download required | [NanoDrone benchmark repository](https://github.com/idsia-robotics/nanodrone-sysid-benchmark) |

## Why some data are not bundled

The CED and Silverbox benchmark pages distribute the data and prescribe citations, but the audit did not find an explicit license granting downstream redistribution of the data files. The NanoDrone authors publicly release the data and code, but no explicit repository license was found at audit time. To avoid implying rights that the project does not hold, those raw files are not redistributed here.

This is a conservative packaging decision, not a claim that reuse is prohibited. Obtain the files from their canonical publishers and follow their terms. The included configurations, trained parameters, figures, and metrics are project outputs; they do not grant rights in, or act as substitutes for, the external raw datasets.

## Restore external data

From the repository root, install the optional benchmark loader and run:

```bash
python -m pip install nonlinear-benchmarks
python datasets/fetch_external_datasets.py all
python datasets/fetch_external_datasets.py verify
```

The fetcher uses only canonical sources, places files in the paths expected by the framework, and checks every downloaded raw file against the SHA-256 and byte size recorded in `EXTERNAL_DATA_MANIFEST.json`. A source update that changes bytes fails visibly instead of being accepted silently.

## Citations

- **CED:** T. Wigren and M. Schoukens, *Coupled Electric Drives Data Set and Reference Models*, Technical Report 2017-024, Uppsala University, 2017.
- **Silverbox:** T. Wigren and J. Schoukens, “Three free data sets for development and benchmarking in nonlinear system identification,” *European Control Conference*, 2013. DOI: `10.23919/ECC.2013.6669201`.
- **NanoDrone:** M. Busetto et al., “Nonlinear system identification for a nano-drone benchmark,” *Control Engineering Practice*, 172, 106871, 2026.
- **Deep dissipative n-link generator:** Y. Okamoto and R. Kojima, “Learning Deep Dissipative Dynamics,” *AAAI Conference on Artificial Intelligence*, 2025.

See `DATASET_AUDIT.md`, `THIRD_PARTY_NOTICES.md`, and `licenses/` for the full attribution and redistribution assessment. Audit date: **2026-09-27**. This record is an engineering provenance review, not legal advice.
