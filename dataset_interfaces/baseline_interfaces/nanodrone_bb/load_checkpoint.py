#!/usr/bin/env python3
# Author: Simone Betteti
"""Load the frozen S3 black-box model and optionally execute one zero rollout."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from models import ResidualQuadModel

DEFAULT = ROOT / "datasets/nanodrone_S3/checkpoints/baselines/s3_bb_T2_seed47_best_physical_mae.pt"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT)
    parser.add_argument("--smoke-rollout", action="store_true")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = ResidualQuadModel(num_layers=5, hidden_dim=int(config.get("width", 128)), dt=0.01)
    model.load_state_dict(state["model_state"])
    model.eval()
    if args.smoke_rollout:
        with torch.no_grad(): model(torch.zeros(1, 12), torch.zeros(1, 1, 4))
    print(f"loaded {args.checkpoint.name}")

if __name__ == "__main__": main()
