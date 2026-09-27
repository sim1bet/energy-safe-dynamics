# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Component-wise Duffing input-radius certificate artifacts."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "EBM_model", ROOT / "Interface_code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Duffing_DoubleWell_main import _build_layers
from Stress_tests.certificate import compute_uniform_radius_certificate
from Stress_tests.boundary import project_to_energy_level
from Stress_tests.config import UncertaintySet
from Stress_tests.integration import apply_runtime_config
from Stress_tests.model_adapter import EBMStressAdapter


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compute_duffing_radius_artifact(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config" / "config.json").read_text())
    audit = json.loads((run_dir / "audits" / "audit.json").read_text())
    with (run_dir / "checkpoints" / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)

    apply_runtime_config(config)
    adapter = EBMStressAdapter(
        params, _build_layers(config), d=int(config["d"]),
        m=int(config["m_ports"]), dt=0.02,
    )
    stress_arrays = np.load(run_dir / "stress_tests" / "arrays" / "stress_arrays.npz")
    boundary_raw = np.asarray(stress_arrays["boundary_states"])
    epsilon = float(audit["metrics"]["stress"]["epsilon"])
    boundary, projected_residual = project_to_energy_level(
        adapter, boundary_raw, epsilon
    )
    all_wells = np.asarray(stress_arrays["minima_states"])
    well_grad_norm = np.linalg.norm(
        np.asarray(adapter.grad_energy_batch(all_wells)), axis=1
    )
    valid_wells = np.flatnonzero(well_grad_norm <= 0.05)
    if len(valid_wells) < min(2, len(all_wells)):
        valid_wells = np.argsort(well_grad_norm)[:min(2, len(all_wells))]
    wells = all_wells[valid_wells]
    if "boundary_anchor_indices" in stress_arrays.files:
        original_components = np.asarray(stress_arrays["boundary_anchor_indices"], dtype=int)
    else:
        original_components = np.argmin(
            np.linalg.norm(boundary[:, None, :] - all_wells[None, :, :], axis=2), axis=1
        )
    keep = np.isin(original_components, valid_wells)
    boundary = boundary[keep]
    projected_residual = projected_residual[keep]
    component_map = {int(original): mapped for mapped, original in enumerate(valid_wells)}
    component_indices = np.asarray(
        [component_map[int(original)] for original in original_components[keep]], dtype=int
    )

    uncertainty_payload = audit["metrics"]["stress"]["config"]["uncertainty"]
    uncertainty = UncertaintySet(**uncertainty_payload)
    radius = compute_uniform_radius_certificate(
        adapter, boundary, component_indices, uncertainty
    )

    out_dir = run_dir / "stress_tests" / "theory_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "radius_certificate.npz"
    json_path = out_dir / "radius_certificate.json"
    np.savez_compressed(
        npz_path,
        boundary_states=boundary,
        boundary_states_before_projection=boundary_raw,
        component_wells=wells,
        component_well_grad_norm=well_grad_norm[valid_wells],
        boundary_energy_residual=projected_residual,
        component_indices=radius.component_indices,
        pointwise_radius=radius.pointwise_radius,
        grad_norm=radius.grad_norm,
        residual_margin=radius.residual_margin,
        input_dual_norm=radius.input_dual_norm,
        normal_dissipation=radius.normal_dissipation,
        normal_input_gain=radius.normal_input_gain,
        normal_disturbance_support=radius.normal_disturbance_support,
    )
    payload = {
        "run_id": run_dir.name,
        "theory_formula": "rho_star = inf_{Gamma, a_H != 0} delta/||a_H||_*",
        "regular_boundary_estimate": "(r_min*kappa-w_perp_max)/g_perp_max",
        "epistemic_status": (
            "Sampled shell estimates of the continuous-boundary extrema; "
            "not interval-certified bounds over unsampled states."
        ),
        "component_summaries": radius.component_summaries,
        "well_filter": {
            "grad_norm_tolerance": 0.05,
            "retained_original_indices": valid_wells.tolist(),
            "retained_grad_norms": well_grad_norm[valid_wells].tolist(),
        },
        "boundary_projection": {
            "max_abs_residual_after": float(np.max(np.abs(projected_residual))),
            "median_abs_residual_after": float(np.median(np.abs(projected_residual))),
        },
        "arrays": str(npz_path),
    }
    json_path.write_text(json.dumps(payload, default=_jsonable, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(compute_duffing_radius_artifact(sys.argv[1]), indent=2))