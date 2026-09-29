#!/usr/bin/env python3
# Author: Simone Betteti
"""Validate frozen HNER99 reports and their recorded checkpoint provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_hner99.artifacts import artifact_index, validate_index_paths

CERT_ROOT = ROOT / "certificates" / "hner99"
PROVENANCE = CERT_ROOT / "FROZEN_RESULTS_PROVENANCE.json"
EXPECTED_REPORTS = 9
VALID_STATUSES = {
    "EMPIRICAL_RUNTIME_HNER",
    "OK",
    "POST_STEP_PROJECTION_DIAGNOSTIC_ONLY",
    "UNRELIABLE_ZERO_INPUT_OUTWARD",
    "UNRELIABLE_NEAR_CRITICAL_SHELL",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    checks: list[dict[str, object]] = []
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    models = provenance.get("models", {})
    index = artifact_index()
    indexed_reports = index.get("reports", {})
    index_errors = validate_index_paths()
    if len(indexed_reports) != EXPECTED_REPORTS:
        errors.append(
            f"artifact index expected {EXPECTED_REPORTS} reports, "
            f"found {len(indexed_reports)}"
        )
    errors.extend(f"artifact index missing path: {item}" for item in index_errors)

    for name, item in sorted(models.items()):
        checkpoint = item.get("contained_checkpoint")
        expected_hash = item.get("checkpoint_sha256") or item.get(
            "contained_checkpoint_sha256"
        )
        result: dict[str, object] = {
            "model": name,
            "identity_status": item.get("identity_status"),
        }
        if checkpoint:
            path = ROOT / checkpoint
            result["checkpoint_exists"] = path.is_file()
            if not path.is_file():
                errors.append(f"missing checkpoint for {name}: {checkpoint}")
            elif expected_hash:
                actual = _sha256(path)
                result["sha256_matches"] = actual == expected_hash
                if actual != expected_hash:
                    errors.append(f"checkpoint hash mismatch for {name}")
        checks.append(result)

    reports = sorted(
        path for path in CERT_ROOT.rglob("*_runtime.json") if path != PROVENANCE
    )
    if len(reports) != EXPECTED_REPORTS:
        errors.append(f"expected {EXPECTED_REPORTS} reports, found {len(reports)}")

    report_checks: list[dict[str, object]] = []
    required = {
        "dataset",
        "model",
        "q",
        "epsilon_q",
        "k_shell_requested",
        "k_shell_accepted",
        "max_shell_error",
        "hner99_min",
        "shell_32_result",
        "shell_64_result",
        "status",
    }
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(ROOT))
        missing = sorted(required - data.keys())
        status = data.get("status")
        accepted = data.get("k_shell_accepted")
        shell_error = data.get("max_shell_error")
        item_errors: list[str] = []
        if missing:
            item_errors.append("missing fields: " + ", ".join(missing))
        if status not in VALID_STATUSES:
            item_errors.append(f"unknown status: {status!r}")
        if not isinstance(accepted, int) or accepted <= 0:
            item_errors.append(f"invalid accepted-shell count: {accepted!r}")
        if not isinstance(shell_error, (int, float)) or shell_error >= 1e-4:
            item_errors.append(f"shell error outside tolerance: {shell_error!r}")
        for key in ("hner99_min", "shell_32_result", "shell_64_result"):
            value = data.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                item_errors.append(f"invalid {key}: {value!r}")
        if item_errors:
            errors.extend(f"{relative}: {message}" for message in item_errors)
        report_checks.append(
            {
                "path": relative,
                "status": status,
                "accepted_shell_states": accepted,
                "max_shell_error": shell_error,
                "validation": "PASS" if not item_errors else "FAIL",
            }
        )

    output = {
        "status": "PASS" if not errors else "FAIL",
        "report_count": len(reports),
        "artifact_index_count": len(indexed_reports),
        "artifact_index_paths_valid": not index_errors,
        "provenance_checks": checks,
        "report_checks": report_checks,
        "documented_gaps": {
            "silverbox_phebm": models.get("silverbox_phebm", {}).get(
                "identity_status"
            ),
            "nanodrone_phebm": models.get("nanodrone_phebm", {}).get(
                "identity_status"
            ),
        },
        "errors": errors,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
