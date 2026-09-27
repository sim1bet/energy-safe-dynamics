#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Aggregate the fitting-suite output, certificate-suite output, and
metadata registries into one machine-readable status summary. This is a
thin aggregator; the authoritative, human-readable narrative lives in
``copilotA1_theory_aligned_stress_suite_report.md`` and
``FORMAL_GUARANTEE_STATUS.md`` (written separately, not generated
verbatim from this script, since those documents require the qualitative
judgment this script deliberately does not attempt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SUITE_ROOT = Path(__file__).resolve().parents[1]


def main(fitting_json: str, certificate_json: str) -> int:
    fitting = json.loads(Path(fitting_json).read_text())
    certificate = json.loads(Path(certificate_json).read_text())
    obligations = yaml.safe_load((SUITE_ROOT / "metadata" / "proof_obligations.yaml").read_text())

    status_counts: dict[str, int] = {}
    for o in obligations["obligations"]:
        status_counts[o["status"]] = status_counts.get(o["status"], 0) + 1

    summary = {
        "fitting": {
            "metric": fitting["metric_name"],
            "value": fitting["test_nrmse"],
            "threshold": fitting["target_threshold"],
            "target_met": fitting["target_met"],
        },
        "certificate_structural_checks": [
            {"id": r["obligation_id"], "passed": r["passed"]}
            for r in certificate["canonical_structural_checks"]
        ],
        "obligation_status_counts": status_counts,
        "total_obligations": len(obligations["obligations"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
