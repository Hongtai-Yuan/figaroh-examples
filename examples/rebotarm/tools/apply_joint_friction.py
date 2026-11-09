#!/usr/bin/env python3
"""Subtract an independently accepted joint-friction model from a SysID log."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    model_path = Path(args.model).resolve()
    model = json.loads(model_path.read_text())
    if model.get("schema") != "rebotarm.joint_friction_model/v1" or not model.get("accepted"):
        raise ValueError("friction model is not independently accepted")
    joint = int(model["joint"]) - 1
    coefficient = model["deployed_coefficients"]
    frame = pd.read_csv(source, float_precision="round_trip")
    t = frame["t"].to_numpy(dtype=float)
    q = frame[[f"q{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t) - dt)) > 1e-6:
        raise ValueError("input must have a uniform clock")
    sos = butter(4, 4.0, btype="lowpass", fs=1.0 / dt, output="sos")
    q_filtered = sosfiltfilt(sos, q, axis=0)
    velocity = np.gradient(q_filtered, dt, axis=0)
    friction = (
        float(coefficient["viscous_nm_per_rad_s"]) * velocity[:, joint]
        + float(coefficient["coulomb_nm"]) * np.tanh(velocity[:, joint] / 0.03)
        + float(coefficient["offset_nm"])
    )
    corrected = frame.copy()
    corrected[f"tau{joint + 1}"] = corrected[f"tau{joint + 1}"] - friction
    output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output, index=False, float_format="%.12g")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.friction_corrected_sysid_log/v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "friction_model": str(model_path),
                "friction_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "corrected_joint": joint + 1,
                "velocity_filter": "4th-order zero-phase Butterworth, 4 Hz",
                "sample_rate_hz": 1.0 / dt,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output}; corrected J{joint+1} friction")


if __name__ == "__main__":
    main()
