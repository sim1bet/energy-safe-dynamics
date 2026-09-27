"""Held-out and structural audit for the n=3 full-matrix PortHNN-u run."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from porthnn_u.three_link import initialize, hamiltonian, matrices, rollout, vector_field
from train_three_link import DATA, DT, canonical

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def load_params(path):
    arrays = np.load(path)
    params = initialize(seed=0, input_dim=1)
    for branch in ("hamiltonian", "j_head", "r_head", "g_head"):
        layers = []
        for index, layer in enumerate(params[branch]):
            replacement = {name: jnp.asarray(arrays[f"{branch}_{index}_{name}"])
                           if f"{branch}_{index}_{name}" in arrays else value
                           for name, value in layer.items()}
            layers.append(replacement)
        params[branch] = tuple(layers)
    return params

def finite(value): return bool(np.all(np.isfinite(value)))
def main():
    global DATA
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument('--data-dir',type=Path,default=DATA); args = parser.parse_args()
    DATA = args.data_dir
    run = args.run_dir; params = load_params(run / "checkpoint_best.npz")
    truth = np.asarray(canonical(np.load(DATA / "ex1_n3nlink.test.state.npy")), dtype=np.float64)
    inputs = np.asarray(np.load(DATA / "ex1_n3nlink.test.input.npy"), dtype=np.float64)
    norm = np.load(run / "normalization.npz"); train_scale = norm["state_std"]
    predict = jax.jit(jax.vmap(lambda x0, u: rollout(params, x0, u[:-1], DT)))
    predicted = np.asarray(predict(jnp.asarray(truth[:, 0]), jnp.asarray(inputs)))
    error = predicted - truth
    # Match the upstream output-only benchmark exactly: terminal Cartesian
    # position, flattened over trajectories/time/output channels, NRMSE divided
    # by the scalar standard deviation of the complete true-output array.
    def tip_output(states):
        q = states[..., :3]
        return np.stack((np.sin(q).sum(axis=-1) / 3., 1. - np.cos(q).sum(axis=-1) / 3.), axis=-1)
    observed_truth = np.load(DATA / "ex1_n3nlink.test.obs.npy").astype(np.float64)
    observed_prediction = tip_output(predicted)
    if observed_truth.shape != observed_prediction.shape:
        raise ValueError(f"output shape mismatch: {observed_truth.shape} != {observed_prediction.shape}")
    output_rmse = float(np.sqrt(np.mean((observed_prediction-observed_truth)**2)))
    output_nrmse = output_rmse / max(float(np.std(observed_truth)), 1e-12)
    per_channel_rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
    # NRMSE is normalized by frozen train-split channel scales, not test scale.
    per_channel_nrmse = per_channel_rmse / train_scale
    aggregate_rmse = float(np.sqrt(np.mean(error**2)))
    aggregate_nrmse = float(np.sqrt(np.mean((error / train_scale)**2)))
    horizon = truth.shape[1]; quarters = np.array_split(np.arange(1, horizon), 4)
    quarter_rmse = [float(np.sqrt(np.mean(error[:, q]**2))) for q in quarters]
    one_step = error[:, 1]
    sample_states = truth.reshape(-1, 6)
    sample_inputs = inputs.reshape(-1, 1)
    # A deterministic capped test-state sample prevents diagnostic OOM.
    selection = np.linspace(0, len(sample_states)-1, min(1024, len(sample_states)), dtype=int)
    xs = jnp.asarray(sample_states[selection]); us = jnp.asarray(sample_inputs[selection])
    grad = jax.vmap(lambda x: jax.grad(hamiltonian, 1)(params, x))(xs)
    j, r, g = jax.vmap(lambda x: matrices(params, x))(xs)
    vf = jax.vmap(lambda x, u: vector_field(params, x, u))(xs, us)
    conservative = jnp.einsum("bij,bj->bi", j, grad)
    dissipative = -jnp.einsum("bij,bj->bi", r, grad)
    input_part = jnp.einsum("bij,bj->bi", g, us)
    d_hdt = jnp.einsum("bi,bi->b", grad, vf)
    power_rhs = -jnp.einsum("bi,bij,bj->b", grad, r, grad) + jnp.einsum("bi,bij,bj->b", grad, g, us)
    singular = np.asarray(jnp.linalg.svd(g, compute_uv=False)); eig = np.asarray(jnp.linalg.eigvalsh(r))
    report = {
        "status": "completed held-out evaluation",
        "checkpoint": "checkpoint_best.npz",
        "data": {"test_split": "ex1_n3nlink.test", "trajectories": int(truth.shape[0]), "steps": int(truth.shape[1]), "dt_seconds": DT,
                 "state": "canonical (q,p), p=M(q)qdot", "input_dim": 1},
        "metrics": {"aggregate_rmse": aggregate_rmse, "aggregate_nrmse_train_scale": aggregate_nrmse,
                    "ph_ebm_comparable_output_rmse": output_rmse, "ph_ebm_comparable_output_nrmse": output_nrmse,
                    "one_step_rmse": float(np.sqrt(np.mean(one_step**2))), "per_channel_rmse": per_channel_rmse.tolist(),
                    "per_channel_nrmse_train_scale": per_channel_nrmse.tolist(), "quarter_rollout_rmse": quarter_rmse,
                    "maximum_state_norm_prediction": float(np.max(np.linalg.norm(predicted, axis=-1))),
                    "nonfinite_prediction_count": int(np.size(predicted)-np.count_nonzero(np.isfinite(predicted)))},
        "structure": {"sample_count": int(len(selection)), "j_skew_max_abs": float(jnp.max(jnp.abs(j+j.swapaxes(-1,-2)))),
                      "r_symmetry_max_abs": float(jnp.max(jnp.abs(r-r.swapaxes(-1,-2)))), "r_min_eigenvalue": float(np.min(eig)),
                      "g_singular_value_min": float(np.min(singular)), "g_singular_value_max": float(np.max(singular)),
                      "g_effective_rank_mean": float(np.mean(np.sum(singular > 1e-8, axis=-1))),
                      "grad_h_norm_mean": float(jnp.mean(jnp.linalg.norm(grad, axis=-1))),
                      "j_grad_h_norm_mean": float(jnp.mean(jnp.linalg.norm(conservative, axis=-1))),
                      "minus_r_grad_h_norm_mean": float(jnp.mean(jnp.linalg.norm(dissipative, axis=-1))),
                      "g_u_norm_mean": float(jnp.mean(jnp.linalg.norm(input_part, axis=-1))),
                      "conservative_power_max_abs": float(jnp.max(jnp.abs(jnp.einsum('bi,bi->b',grad,conservative)))),
                      "dissipative_power_max": float(jnp.max(-jnp.einsum('bi,bij,bj->b',grad,r,grad))),
                      "power_balance_residual_max_abs": float(jnp.max(jnp.abs(d_hdt-power_rhs))),
                      "finite_diagnostics": finite(np.asarray(vf))},
    }
    np.savez_compressed(run / "test_predictions.npz", truth=truth, prediction=predicted, inputs=inputs, error=error,
                        output_truth=observed_truth, output_prediction=observed_prediction)
    write(run / "test_evaluation.json", report); print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
