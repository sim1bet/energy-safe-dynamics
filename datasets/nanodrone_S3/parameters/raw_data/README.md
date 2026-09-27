# NanoDrone S3 raw data

The raw flight-log CSV files are intentionally not redistributed because the audit did not find an explicit license in the official upstream repository. Retrieve and verify them with:

```bash
python datasets/fetch_external_datasets.py nanodrone
```

The command creates the expected `data/train` and `data/test` directories. Canonical source and citation details are in `../../../EXTERNAL_DATA_MANIFEST.json` and `../../../README.md`.
