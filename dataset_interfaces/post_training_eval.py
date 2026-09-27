"""post_training_eval.py — shared post-training evaluation pipeline.

Both experiments (``duffing_doublewell``, ``deep_dissipative_nlink``) call
into this module instead of duplicating reconstruction-metric/plotting/
Stress_tests-invocation logic. Experiment-specific behaviour (e.g. Duffing's
physical phase portraits) is injected via the ``extra_plot_fn`` callback
rather than an ``if experiment == ...`` branch here.

Pipeline (prompt §11/§24):
    1. reconstruction metrics + signal plots (+ underlying arrays saved)
    2. experiment-specific trajectory/Hamiltonian plots (via callback)
    3. Stress_tests suite (reuses Stress_tests.suite.run_stress_suite unchanged)
    4. audit.json + run_status.json + printed run summary

Each stage is wrapped so a failure in one (e.g. a plotting exception) does not
abort the others or discard already-computed metrics (prompt §23).
"""
from __future__ import annotations

import json
import pickle
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_paths import ExperimentPaths, get_experiment_paths
from plot_utils import plot_true_vs_pred

STATUS_KEYS = (
    "training_complete", "reconstruction_evaluation_complete", "signal_plots_complete",
    "trajectory_plots_complete", "stress_tests_complete", "stress_plots_complete",
    "theory_aligned_certificates_complete", "audit_complete",
)

# Stress_tests/theory_aligned/certificates + fitting are plain, sys.path-based
# modules (no package __init__.py), matching how Stress_tests/theory_aligned/
# scripts/*.py already import them -- mirrored here so the checks run inline
# during post_training_eval instead of only via manual, standalone invocation.
_THEORY_ALIGNED_DIR = _ROOT / "Stress_tests" / "theory_aligned"


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(path: str | Path, **fields) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, default=_json_default))


def write_run_status(path: str | Path, **flags) -> None:
    complete = {k: bool(flags.get(k, False)) for k in STATUS_KEYS}
    save_json(path, **complete)


def save_checkpoint(path: str | Path, params) -> None:
    """Minimal params-pytree checkpoint (no orbax/flax dependency in this repo)."""
    import jax
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(path, "wb") as f:
        pickle.dump(numpy_params, f)


def load_checkpoint(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _per_trajectory_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    err = y_true - y_pred
    return np.sqrt(np.mean(err.reshape(err.shape[0], -1) ** 2, axis=-1))


def evaluate_reconstruction(
    y_true: np.ndarray, y_pred: np.ndarray, *, u: Optional[np.ndarray] = None,
    dt: Optional[float] = None, unit_label: str = "", paths: ExperimentPaths = None,
    prefix: str = "test", max_points: int = 5000, extra_fixed_indices: tuple = (0,),
) -> dict:
    """Reconstruction metrics + best/median/worst/fixed representative plots
    (prompt §16-17). ``y_true``/``y_pred``: ``[N_trajectories, T, C]``.

    Saves plots under ``paths.signals_dir`` and the underlying numerical
    arrays (time/input/truth/prediction/error) alongside them as ``.npz``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"Prediction/target mismatch: pred={y_pred.shape}, target={y_true.shape}"
    N = y_true.shape[0]

    per_traj = _per_trajectory_rmse(y_true, y_pred)
    overall_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    overall_nrmse = overall_rmse / max(float(np.std(y_true)), 1e-12)
    overall_mae = float(np.mean(np.abs(y_true - y_pred)))

    order = np.argsort(per_traj)
    selected = {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }
    for i in extra_fixed_indices:
        if 0 <= i < N:
            selected[f"fixed_{i}"] = int(i)

    signals_dir = paths.signals_dir if paths is not None else Path(".")
    signals_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = {}
    for tag, idx in selected.items():
        yt, yp = y_true[idx], y_pred[idx]
        uu = None if u is None else np.asarray(u)[idx]
        rmse_i, nrmse_i = float(per_traj[idx]), float(per_traj[idx] / max(np.std(yt), 1e-12))
        png_path = signals_dir / f"{prefix}_{tag}_traj{idx}.png"
        try:
            plot_true_vs_pred(
                yt, yp, str(png_path), dt=dt, unit_label=unit_label,
                title=f"{prefix} [{tag}] traj {idx}  RMSE={rmse_i:.4g}  NRMSE={nrmse_i:.4g}",
                rmse=rmse_i, nrmse=nrmse_i, max_points=max_points, u_input=uu,
            )
            saved_plots[tag] = str(png_path)
        except Exception as exc:  # noqa: BLE001 -- one bad plot must not abort evaluation
            print(f"[post_training_eval] plot '{tag}' (traj {idx}) failed: {exc}")
        npz_path = signals_dir / f"{prefix}_{tag}_traj{idx}.npz"
        t = np.arange(yt.shape[0]) * (dt or 1.0)
        np.savez_compressed(npz_path, time=t, ground_truth=yt, prediction=yp,
                           reconstruction_error=yt - yp, **({"input": uu} if uu is not None else {}))

    return dict(
        rmse=overall_rmse, nrmse=overall_nrmse, mae=overall_mae,
        per_trajectory_rmse=per_traj.tolist(),
        selected_trajectories=selected, saved_plots=saved_plots,
    )


def run_stress_tests(
    *, params, layers, d: int, m: int, dt: float, epsilon: float, gamma: float,
    reference_states: np.ndarray, reference_inputs: np.ndarray,
    output_dir: str | Path, uncertainty=None, stress_config=None, seed: int = 0,
) -> dict:
    """Thin, unmodified reuse of ``Stress_tests.suite.run_stress_suite`` --
    no certificate/geometry mathematics is duplicated here."""
    from Stress_tests.config import StressTestConfig, UncertaintySet
    from Stress_tests.suite import run_stress_suite

    cfg = stress_config
    if cfg is None:
        cfg = StressTestConfig(seed=seed, uncertainty=uncertainty or UncertaintySet())
    return run_stress_suite(
        params=params, layers=layers, d=d, m=m, dt=dt, epsilon=epsilon, gamma=gamma,
        reference_states=reference_states, reference_inputs=reference_inputs,
        output_dir=output_dir, config=cfg,
    )


def run_theory_aligned_certificates(checkpoint_path: str | Path, config_path: str | Path,
                                    stress_arrays_npz: str | Path, output_dir: str | Path) -> dict:
    """Run the canonical, theory-aligned certificate checks (see
    Stress_tests/theory_aligned/README.md) against a just-saved checkpoint.

    Application-agnostic: works for any experiment whose checkpoint is an
    EBMParams pytree produced by the shared EBM_model package. JAX-free
    (numpy + pyyaml only, via certificates/checkpoint_io.py's custom
    unpickler), so it runs inline here even though this module is otherwise
    called from inside a JAX training process. Never raises; per-check
    failures are recorded in the returned dict's ``errors`` field instead of
    aborting the caller (same guarding discipline as every other stage of
    ``run_post_training_evaluation``).
    """
    for _p in (_THEORY_ALIGNED_DIR / "certificates", _THEORY_ALIGNED_DIR / "fitting"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"structural_checks": None, "unforced_dissipativity_probe": None,
                    "equilibrium_hessian_probe": None}
    errors: dict = {}

    try:
        from structural_checks import run_all_structural_checks
        from dataclasses import asdict
        result["structural_checks"] = [asdict(r) for r in run_all_structural_checks(str(checkpoint_path), str(config_path))]
    except Exception as exc:  # noqa: BLE001
        errors["structural_checks"] = f"{exc}"
        print(f"[post_training_eval] theory-aligned structural checks FAILED:\n{traceback.format_exc()}")

    try:
        from dataclasses import asdict
        from unforced_dissipativity_probe import run_probe as run_dissipativity_probe
        result["unforced_dissipativity_probe"] = asdict(run_dissipativity_probe(str(checkpoint_path), str(config_path)))
    except Exception as exc:  # noqa: BLE001
        errors["unforced_dissipativity_probe"] = f"{exc}"
        print(f"[post_training_eval] theory-aligned dissipativity probe FAILED:\n{traceback.format_exc()}")

    if Path(stress_arrays_npz).exists():
        try:
            from dataclasses import asdict
            from equilibrium_hessian_probe import run_probe as run_equilibrium_probe
            result["equilibrium_hessian_probe"] = asdict(
                run_equilibrium_probe(str(checkpoint_path), str(config_path), str(stress_arrays_npz)))
        except Exception as exc:  # noqa: BLE001
            errors["equilibrium_hessian_probe"] = f"{exc}"
            print(f"[post_training_eval] theory-aligned equilibrium/Hessian probe FAILED:\n{traceback.format_exc()}")
    else:
        errors["equilibrium_hessian_probe"] = (
            f"skipped: {stress_arrays_npz} does not exist (legacy Stress_tests suite did not "
            "complete far enough to save minima_states, e.g. boundary-tracing failure)"
        )

    result["errors"] = errors
    save_json(output_dir / "certificate_results.json", **result)
    return result


def print_run_summary(experiment_name: str, run_id: str, metrics: dict, errors: dict,
                      status: dict, paths: ExperimentPaths) -> None:
    recon = metrics.get("reconstruction", {})
    stress = metrics.get("stress", {})
    cert = stress.get("certificate_boundary_audit", {}) if isinstance(stress, dict) else {}
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Run: {run_id}")
    print()
    print("Test reconstruction:")
    print(f"  RMSE:  {recon.get('rmse')}")
    print(f"  NRMSE: {recon.get('nrmse')}")
    print()
    print("Stress testing:")
    print(f"  epsilon: {stress.get('epsilon', metrics.get('epsilon'))}")
    print(f"  min robust implemented slack: {cert.get('min_robust_implemented_slack')}")
    print(f"  formula violation fraction: {cert.get('formula_violation_fraction')}")
    print(f"  max formula/implementation gap: {cert.get('max_formula_vs_implemented_hdot_gap_at_formula_input')}")
    print()
    print("Outputs:")
    print(f"  results root: {paths.run_root}")
    print(f"  reconstruction plots: {paths.signals_dir}")
    print(f"  stress plots: {paths.stress_test_dir}")
    print(f"  audit file: {paths.audits_dir / 'audit.json'}")
    if errors:
        print()
        print("Stages with errors (see run_status.json / logs above):")
        for k, v in errors.items():
            print(f"  {k}: {v}")
    print("=" * 60)


def run_post_training_evaluation(
    *, experiment_name: str, run_id: str,
    params, layers, grad_E, d: int, m: int, dt: float,
    test_true: np.ndarray, test_pred: np.ndarray, test_u: Optional[np.ndarray] = None,
    unit_label: str = "",
    reference_states: np.ndarray, reference_inputs: np.ndarray,
    epsilon: float, gamma: float, uncertainty=None, stress_config=None,
    paths: Optional[ExperimentPaths] = None, extra_plot_fn: Optional[Callable] = None,
    extra_metrics: Optional[dict] = None, seed: int = 0,
) -> dict:
    """The common post-training evaluator (prompt §24): reconstruction ->
    experiment-specific plots -> Stress_tests -> audit/status/summary.
    Never raises -- every stage is independently guarded (prompt §23)."""
    if paths is None:
        paths = get_experiment_paths(experiment_name, run_id)

    status = {k: False for k in STATUS_KEYS}
    status["training_complete"] = True
    metrics: dict = dict(extra_metrics or {})
    errors: dict = {}

    try:
        metrics["reconstruction"] = evaluate_reconstruction(
            test_true, test_pred, u=test_u, dt=dt, unit_label=unit_label, paths=paths, prefix="test")
        status["reconstruction_evaluation_complete"] = True
        status["signal_plots_complete"] = True
    except Exception as exc:  # noqa: BLE001
        errors["reconstruction"] = f"{exc}"
        print(f"[post_training_eval] reconstruction evaluation FAILED:\n{traceback.format_exc()}")

    if extra_plot_fn is not None:
        try:
            extra_plot_fn(paths)
            status["trajectory_plots_complete"] = True
        except Exception as exc:  # noqa: BLE001
            errors["trajectory_plots"] = f"{exc}"
            print(f"[post_training_eval] experiment-specific trajectory/Hamiltonian plots FAILED:\n{traceback.format_exc()}")
    else:
        status["trajectory_plots_complete"] = True  # nothing requested

    try:
        metrics["stress"] = run_stress_tests(
            params=params, layers=layers, d=d, m=m, dt=dt, epsilon=epsilon, gamma=gamma,
            reference_states=reference_states, reference_inputs=reference_inputs,
            output_dir=paths.stress_test_dir, uncertainty=uncertainty, stress_config=stress_config, seed=seed,
        )
        metrics["epsilon"] = epsilon
        metrics["gamma"] = gamma
        status["stress_tests_complete"] = True
        status["stress_plots_complete"] = True
    except Exception as exc:  # noqa: BLE001
        errors["stress_tests"] = f"{exc}"
        print(f"[post_training_eval] Stress_tests suite FAILED:\n{traceback.format_exc()}")

    try:
        metrics["theory_aligned_certificates"] = run_theory_aligned_certificates(
            checkpoint_path=paths.checkpoint_dir / "params.pkl",
            config_path=paths.config_dir / "config.json",
            stress_arrays_npz=paths.stress_test_dir / "arrays" / "stress_arrays.npz",
            output_dir=paths.stress_test_dir / "theory_aligned",
        )
        status["theory_aligned_certificates_complete"] = True
    except Exception as exc:  # noqa: BLE001
        errors["theory_aligned_certificates"] = f"{exc}"
        print(f"[post_training_eval] theory-aligned certificate suite FAILED:\n{traceback.format_exc()}")

    try:
        save_json(paths.audits_dir / "audit.json", experiment_name=experiment_name, run_id=run_id,
                 seed=seed, epsilon=epsilon, gamma=gamma, metrics=metrics, errors=errors)
        status["audit_complete"] = True
    except Exception as exc:  # noqa: BLE001
        errors["audit"] = f"{exc}"
        print(f"[post_training_eval] audit save FAILED: {exc}")

    try:
        sys.path.insert(0, str(_THEORY_ALIGNED_DIR / "fitting"))
        from fitting_report import build_fitting_report
        fitting = build_fitting_report(str(paths.run_root), target_threshold=0.05)
        save_json(paths.stress_test_dir / "theory_aligned" / "fitting_evidence.json", **fitting.__dict__)
        if experiment_name == "duffing_doublewell":
            from Stress_tests.theory_aligned.duffing_radius_certificate import compute_duffing_radius_artifact
            from Stress_tests.theory_aligned.duffing_certificate_figure import render_duffing_certificate_figure
            metrics["theory_aligned_radius"] = compute_duffing_radius_artifact(paths.run_root)
            metrics["theory_aligned_figure"] = render_duffing_certificate_figure(paths.run_root)
        elif experiment_name == "deep_dissipative_nlink":
            from Stress_tests.theory_aligned.nlink_certificate_figure import render_nlink_theory_figure
            metrics["theory_aligned_figure"] = render_nlink_theory_figure(paths.run_root)
    except Exception as exc:  # noqa: BLE001
        print(f"[post_training_eval] theory-aligned fitting-evidence snapshot FAILED: {exc}")

    write_run_status(paths.run_root / "run_status.json", **status)
    print_run_summary(experiment_name, run_id, metrics, errors, status, paths)
    return dict(metrics=metrics, errors=errors, status=status, paths=paths)
