#!/usr/bin/env python3
"""Execute short finite inference rollouts for every bundled benchmark."""
from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "EBM_model", ROOT / "dataset_interfaces"): sys.path.insert(0, str(path))
import jax
import jax.numpy as jnp
import numpy as np
import EBM_param_fields as pf
from EBM_param_fields import init_params, make_grad_energy
from EBM_rollout import make_rollout_fn

# EBM_param_fields stores model options as module globals.  Restore every
# option for each champion so an enabled feature cannot leak into the next
# dataset's validation.
PF_DEFAULTS = {name: getattr(pf, name) for name in (
    "CHOL_CLIP_EXP", "DAMPING_SCALE", "B_INIT_SCALE", "VF_SCALE",
    "USE_FEEDTHROUGH", "USE_NONLINEAR_READOUT", "USE_STATE_DAMPING",
    "USE_SATURATING_READOUT", "USE_INPUT_GAIN", "USE_SATURATING_INPUT",
    "USE_INPUT_AWARE_ENCODER", "GRAD_E_CLIP", "XDOT_CLIP", "X0_NORM_MAX",
    "STATE_NORM_CLIP", "USE_IDENTITY_READOUT", "USE_STATE_INTERCONNECTION",
    "USE_QUADRATIC_INTERCONNECTION", "USE_CUBIC_INTERCONNECTION",
    "USE_RICH_INTERCONNECTION", "USE_RICH_DAMPING", "USE_RICH_INPUT_MATRIX",
    "PH_FIELD_WIDTH", "INPUT_MATRIX_STRUCTURE",
)}

def load_config(path, defaults):
    payload = json.loads(Path(path).read_text()); defaults.update(payload.get("config", payload)); return defaults

def apply_flags(cfg):
    mapping = {"CHOL_CLIP_EXP":"chol_clip_exp", "DAMPING_SCALE":"damping_scale", "B_INIT_SCALE":"b_init_scale",
      "VF_SCALE":"vf_scale", "USE_FEEDTHROUGH":"use_feedthrough", "USE_NONLINEAR_READOUT":"use_nonlinear_readout",
      "USE_STATE_DAMPING":"use_state_damping", "USE_SATURATING_READOUT":"use_saturating_readout",
      "USE_INPUT_GAIN":"use_input_gain", "USE_SATURATING_INPUT":"use_saturating_input",
      "USE_INPUT_AWARE_ENCODER":"use_input_aware_encoder", "GRAD_E_CLIP":"grad_e_clip", "XDOT_CLIP":"xdot_clip",
      "X0_NORM_MAX":"x0_norm_max", "STATE_NORM_CLIP":"state_norm_clip", "USE_IDENTITY_READOUT":"use_identity_readout",
      "USE_STATE_INTERCONNECTION":"use_state_interconnection", "USE_QUADRATIC_INTERCONNECTION":"use_quadratic_interconnection",
      "USE_CUBIC_INTERCONNECTION":"use_cubic_interconnection", "USE_RICH_INTERCONNECTION":"use_rich_interconnection",
      "USE_RICH_DAMPING":"use_rich_damping", "USE_RICH_INPUT_MATRIX":"use_rich_input_matrix",
      "PH_FIELD_WIDTH":"ph_field_width", "INPUT_MATRIX_STRUCTURE":"input_matrix_structure"}
    for attr, key in mapping.items():
        setattr(pf, attr, cfg.get(key, PF_DEFAULTS[attr]))

def execute(label, module, cfg, inputs, checkpoint=None, x0=None, legacy=False):
    apply_flags(cfg); layers = module._build_layers(cfg); grad = make_grad_energy(layers)
    d, m = int(cfg["d"]), int(cfg.get("m_ports", inputs.shape[-1]))
    template = init_params(jax.random.PRNGKey(0), d=d, m=m, n_obs=int(cfg.get("n_obs", d)),
        init_win=int(cfg.get("init_win", 1)), layer_dims=list(cfg["layer_dims"]), hidden=int(cfg.get("hidden", 64)), dt=float(cfg.get("dt", 1.0)))
    params, source = template, "initialized"
    if checkpoint:
        if legacy:
            from checkpoint_compat import load_legacy_checkpoint
            params = load_legacy_checkpoint(checkpoint, template)
        else:
            with Path(checkpoint).open("rb") as stream: params = pickle.load(stream)
            if isinstance(params, dict) and "params" in params:
                params = params["params"]
            params = pf.upgrade_legacy_params(params, template)
        source = Path(checkpoint).relative_to(ROOT).as_posix()
    rollout = make_rollout_fn(grad, d, m, float(cfg.get("dt", 1.0)), integrator=str(cfg.get("rollout_integrator", "rk4")), substeps=int(cfg.get("rollout_substeps", 1)))
    arrays = rollout(params, jnp.asarray(np.zeros(d, np.float32) if x0 is None else x0, dtype=jnp.float32), jnp.asarray(inputs[:4], dtype=jnp.float32))
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays): raise FloatingPointError(f"{label}: non-finite inference")
    return {"status":"PASS", "parameter_source":source, "output_shapes":[list(np.asarray(value).shape) for value in arrays]}

def main():
    report = {"datasets": {}}
    try:
        import Duffing_DoubleWell_main as duffing
        cfg = load_config(ROOT/"datasets/duffing_doublewell/champion_configuration/config.json", duffing.get_config())
        data = duffing.load_duffing_doublewell(ROOT/"datasets/duffing_doublewell/parameters")
        cfg["dt"] = float(data.get("_gen_config", {}).get("dt", 0.02))
        report["datasets"]["duffing_doublewell"] = execute("duffing", duffing, cfg, data["u_test"][0], ROOT/"datasets/duffing_doublewell/checkpoints/params.pkl", data["z_test"][0,0])
        import DeepDissipative_NLink_main as nlink
        for suffix, prefix in (("n2","nlink2_100"),("n3","ex1_n3")):
            base=ROOT/f"datasets/deep_dissipative_nlink_{suffix}"; cfg=load_config(base/"champion_configuration/config.json",nlink.get_config()); data=nlink.load_nlink_dataset(base/"parameters",prefix)
            cfg.update(dt=float(data["dt"]),m_ports=int(data["in_dim"]),n_obs=int(data["n_obs"]))
            report["datasets"][f"deep_dissipative_nlink_{suffix}"]=execute(suffix,nlink,cfg,data["u_test"][0],base/"checkpoints/params.pkl")
        import NanoDrone_main as nano
        cfg=load_config(ROOT/"datasets/nanodrone_S3/champion_configuration/s3_ph_T2_seed47.json",nano.get_config())
        from nanodrone.data import load_trajectory, transform_inputs
        trajectory=load_trajectory(next((ROOT/"datasets/nanodrone_S3/parameters/raw_data/data/test").glob("*.csv"))); inputs=transform_inputs(trajectory.u,cfg.get("nanodrone_input_representation","motor_speed")); cfg["dt"]=nano.DT
        report["datasets"]["nanodrone_S3"]=execute("nanodrone",nano,cfg,inputs,ROOT/"datasets/nanodrone_S3/checkpoints/best_physical_mae.pkl",trajectory.y[0])
        import CED_main as ced, Silverbox_main as silver
        cfg=load_config(ROOT/"datasets/CED/champion_configuration/config.json",silver.get_config()); dt,_,tests,_,_=ced._load_ced_raw(); cfg["dt"]=float(dt)
        report["datasets"]["CED"]=execute("CED",silver,cfg,tests[0][0],ROOT/"datasets/CED/checkpoints/params.pkl",legacy=True)
        cfg=load_config(ROOT/"datasets/Silverbox/champion_configuration/config.json",silver.get_config()); u,_,dt,*_=silver.load_silverbox(); cfg["dt"]=float(dt)
        report["datasets"]["Silverbox"]=execute("Silverbox",silver,cfg,u)
        report["status"]="PASS"
    except Exception as exc: report.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(report["status"]!="PASS")

if __name__ == "__main__": main()
