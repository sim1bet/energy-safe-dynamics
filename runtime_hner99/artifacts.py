# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Access the canonical frozen HNER99 artifacts shipped with the release."""
from __future__ import annotations
import json
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_ROOT = RELEASE_ROOT / "certificates" / "hner99"
INDEX_PATH = Path(__file__).with_name("ARTIFACT_INDEX.json")

def artifact_index() -> dict:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))

def provenance() -> dict:
    path = CERTIFICATE_ROOT / "FROZEN_RESULTS_PROVENANCE.json"
    return json.loads(path.read_text(encoding="utf-8"))

def load_frozen_report(selector: str) -> dict:
    reports = artifact_index()["reports"]
    if selector not in reports:
        available = ", ".join(sorted(reports))
        raise KeyError(f"unknown frozen report {selector!r}; available: {available}")
    path = RELEASE_ROOT / reports[selector]["json"]
    return json.loads(path.read_text(encoding="utf-8"))

def validate_index_paths() -> list[str]:
    missing = []
    for selector, item in artifact_index()["reports"].items():
        for kind in ("json", "text"):
            if not (RELEASE_ROOT / item[kind]).is_file():
                missing.append(f"{selector}:{kind}:{item[kind]}")
    return missing

