# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Fixed-budget audit of HNER99 model loaders and shell reconstruction."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .runtime import LOADERS, input_moment, reconstruct_shell, vectorized

RELEASE_ROOT = Path(__file__).resolve().parents[1]

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def audit_model(selector: str) -> dict:
    if selector not in LOADERS:
        raise KeyError(f"unsupported selector {selector!r}")
    model = LOADERS[selector]()
    factor, eigenvalues, rank = input_moment(model.inputs)
    shells = {}
    for count in (32, 64):
        epsilon, points, max_error = reconstruct_shell(model, count)
        shells[str(count)] = {"epsilon_q": epsilon, "accepted": len(points),
                              "max_shell_error": max_error,
                              "finite": bool(np.isfinite(points).all())}
    energies = vectorized(model.energy, model.states)
    passed = all(shell["accepted"] >= 8 and shell["max_shell_error"] < 1e-4
                 and shell["finite"] for shell in shells.values())
    return {"selector": selector, "dataset": model.dataset, "model": model.name,
            "checkpoint": str(model.checkpoint.relative_to(RELEASE_ROOT)),
            "checkpoint_sha256": _sha256(model.checkpoint),
            "n_states": len(model.states), "n_inputs": len(model.inputs),
            "finite_states": bool(np.isfinite(model.states).all()),
            "finite_inputs": bool(np.isfinite(model.inputs).all()),
            "finite_energies": bool(np.isfinite(energies).all()),
            "state_dtype": str(model.states.dtype), "input_active_rank": rank,
            "input_moment_eigenvalues": eigenvalues.tolist(),
            "input_factor_shape": list(factor.shape), "shells": shells,
            "status": "PASS" if passed else "FAIL"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selectors", nargs="*", choices=sorted(LOADERS))
    parser.add_argument("--output", type=Path,
                        default=RELEASE_ROOT / "results/hner99_runtime/audits")
    args = parser.parse_args()
    selectors = args.selectors or sorted(LOADERS)
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for selector in selectors:
        report = audit_model(selector)
        destination = args.output / f"{selector}_audit.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{selector}: {report['status']} -> {destination}")
        failed |= report["status"] != "PASS"
    raise SystemExit(failed)

if __name__ == "__main__":
    main()

