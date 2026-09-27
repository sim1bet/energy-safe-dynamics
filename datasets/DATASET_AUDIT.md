# Dataset attribution and redistribution audit

Audit date: 2026-09-27

## Scope and method

The audit inventoried raw numeric data, derived arrays, configurations, checkpoints, metrics, and figures. For externally sourced material, the canonical project or benchmark site, requested scholarly citation, and an explicit redistribution license were checked separately. A software license was not treated as a license for third-party benchmark data unless the source explicitly said so.

## Decisions

### CED — download-only

The canonical benchmark page supplies `DATAUNIF.MAT` and requests citation of the Uppsala technical report. No explicit dataset redistribution license was identified. The raw MAT file was therefore removed from the release folder. The project configuration and trained parameter file remain.

### Silverbox — download-only

The canonical benchmark page supplies the Silverbox files and requests citation of Wigren and Schoukens (ECC 2013). No explicit dataset redistribution license was identified. The raw MAT/CSV files and their duplicated upstream README files were therefore removed. The project configuration remains.

### NanoDrone S3 — download-only

The official benchmark repository publicly supplies the flight logs, and the associated paper describes the release as open source. No explicit license file was found in the upstream repository during this audit. Because “publicly accessible” does not by itself specify redistribution terms, the raw CSV logs were removed. Project-authored configurations, learned parameters, analyses, and figures remain.

### Deep dissipative n-link — retained with attribution

The n-link arrays were generated through the DeepDissipativeModel benchmark workflow. That upstream generator is distributed under the MIT License. The generated arrays and project outputs are retained, with the upstream MIT notice and scholarly citation recorded here. This conclusion applies to the identified generator code; users should still cite the associated paper.

### Duffing double well — retained

The Duffing samples are synthetic, project-generated artifacts rather than a copied third-party dataset. They are retained with their configurations, learned parameters, and validation results.

## Risk controls

- No external raw file with unresolved redistribution terms is included.
- Every excluded raw file has a canonical target path, byte size, and SHA-256 digest in `EXTERNAL_DATA_MANIFEST.json`.
- Acquisition is explicit: there is no import-time or inference-time network fallback.
- Downloads come from canonical publishers and are accepted only if their recorded digest matches.
- Software licenses for acquisition or generation tools are reproduced separately and are not presented as blanket licenses for benchmark data.

## Limitations

This audit assesses the release contents and source information available on the audit date. It does not determine ownership, database rights, privacy obligations, or whether a particular jurisdiction permits redistribution. If an upstream publisher later adds a clear license, the packaging decision can be revisited and the manifest hashes should be regenerated from that licensed release.
