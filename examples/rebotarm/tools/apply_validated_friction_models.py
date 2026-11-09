#!/usr/bin/env python3
"""Subtract six independently validated friction models from a SysID log."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transition-width", type=float, default=0.03)
    args = parser.parse_args()
    if len(args.model) != 6:
        parser.error("exactly six --model arguments are required")
    if not 0.005 <= args.transition_width <= 0.10:
        parser.error("--transition-width must be in [0.005, 0.10] rad/s")

    input_path = Path(args.input).resolve()
    output = Path(args.output).resolve()
    model_paths = [Path(value).resolve() for value in args.model]
    frame = pd.read_csv(input_path)
    q_columns = [f"q{i}" for i in range(1, 7)]
    tau_columns = [f"tau{i}" for i in range(1, 7)]
    if any(column not in frame for column in ["t", *q_columns, *tau_columns]):
        raise ValueError("input must contain t,q1..q6,tau1..tau6")
    t = frame["t"].to_numpy(float)
    if np.any(np.diff(t) <= 0) or abs(float(np.median(np.diff(t))) - 0.01) > 1e-5:
        raise ValueError("input must have strictly increasing 100 Hz timestamps")
    q = frame[q_columns].to_numpy(float)
    filtered_q = sosfiltfilt(butter(4, 4.0, fs=100.0, output="sos"), q, axis=0)
    velocity = np.gradient(filtered_q, t, axis=0)

    coefficients = []
    provenance = []
    for expected_joint, path in enumerate(model_paths, start=1):
        model = json.loads(path.read_text())
        if (
            model.get("schema") != "rebotarm.joint_friction_model/v1"
            or not model.get("accepted")
            or model.get("joint") != expected_joint
        ):
            raise ValueError(f"{path}: expected accepted joint {expected_joint} model")
        coefficient = model["deployed_coefficients"]
        values = np.array(
            [
                coefficient["viscous_nm_per_rad_s"],
                coefficient["coulomb_nm"],
                coefficient["offset_nm"],
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or values[0] < 0 or values[1] < 0:
            raise ValueError(f"{path}: non-finite or non-dissipative friction coefficient")
        coefficients.append(values)
        provenance.append({"file": str(path), "sha256": digest(path)})

    coefficients = np.asarray(coefficients)
    friction = (
        velocity * coefficients[:, 0]
        + np.tanh(velocity / args.transition_width) * coefficients[:, 1]
        + coefficients[:, 2]
    )
    corrected = frame.copy()
    corrected[tau_columns] = frame[tau_columns].to_numpy(float) - friction
    output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output, index=False)
    metadata = {
        "schema": "rebotarm.friction_corrected_sysid_log/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "source_sha256": digest(input_path),
        "friction_models": provenance,
        "transition_width_rad_s": args.transition_width,
        "coefficients_fv_fc_offset": coefficients.tolist(),
        "output_sha256": digest(output),
        "position_filter": "zero-phase Butterworth order 4, cutoff 4 Hz",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("wrote", output)
    print("max absolute correction", np.max(np.abs(friction), axis=0))


if __name__ == "__main__":
    main()
