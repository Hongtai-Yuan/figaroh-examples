#!/usr/bin/env python3
"""Fit and independently validate one joint's low-speed friction model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.signal import butter, sosfiltfilt


def load(path: Path, model: pin.Model, joint: int):
    frame = pd.read_csv(path)
    t = frame["t"].to_numpy(dtype=float)
    q = frame[[f"q{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    tau = frame[[f"tau{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    # The integrated motor encoders are quantised, especially on joints 4--6.
    # Differentiating raw positions twice falsely labels constant-speed plateaus
    # as high acceleration.  Use the same zero-phase filtering policy as the
    # dynamic identification pipeline before differentiating.
    q_filtered = sosfiltfilt(butter(4, 4.0, fs=100.0, output="sos"), q, axis=0)
    velocity = np.gradient(q_filtered, t, axis=0)
    acceleration = np.gradient(velocity, t, axis=0)
    data = model.createData()
    gravity = np.asarray(
        [pin.computeGeneralizedGravity(model, data, position).copy() for position in q]
    )
    residual = tau[:, joint] - gravity[:, joint]
    mask = (np.abs(velocity[:, joint]) > 0.04) & (
        np.abs(acceleration[:, joint]) < 0.25
    )
    X = np.column_stack(
        (velocity[mask, joint], np.sign(velocity[mask, joint]), np.ones(mask.sum()))
    )
    return X, residual[mask]


def fit(X: np.ndarray, y: np.ndarray):
    coefficient = np.linalg.lstsq(X, y, rcond=None)[0]
    prediction = X @ coefficient
    return coefficient, float(np.sqrt(np.mean((prediction - y) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    urdf, train, validation = map(
        lambda value: Path(value).resolve(), (args.urdf, args.train, args.validation)
    )
    output = Path(args.output).resolve()
    model = pin.buildModelFromUrdf(str(urdf))
    joint = args.joint - 1
    X_train, y_train = load(train, model, joint)
    X_validation, y_validation = load(validation, model, joint)
    if len(y_train) < 300 or len(y_validation) < 300:
        raise RuntimeError("insufficient steady bidirectional samples")
    train_coefficient, train_rmse = fit(X_train, y_train)
    validation_coefficient, validation_refit_rmse = fit(X_validation, y_validation)
    validation_prediction = X_validation @ train_coefficient
    validation_rmse = float(
        np.sqrt(np.mean((validation_prediction - y_validation) ** 2))
    )
    # Viscous friction is accepted only if it is larger than repeatability
    # noise and agrees across splits.  Otherwise deploy zero rather than a
    # large, cancellation-prone coefficient.
    fv_repeatable = (
        train_coefficient[0] >= 0.05
        and validation_coefficient[0] >= 0.0
        and abs(validation_coefficient[0] - train_coefficient[0])
        <= 0.25 * train_coefficient[0]
    )
    deployed = train_coefficient.copy()
    if not fv_repeatable:
        deployed[0] = 0.0
    fs_relative_difference = float(
        abs(validation_coefficient[1] - train_coefficient[1])
        / max(abs(train_coefficient[1]), 1e-9)
    )
    offset_difference = float(abs(validation_coefficient[2] - train_coefficient[2]))
    accepted = bool(
        0.05 <= train_coefficient[1] <= 5.0
        and fs_relative_difference <= 0.10
        and offset_difference <= 0.05
        and validation_rmse <= 0.30
        and np.linalg.cond(X_train) <= 100.0
    )
    result = {
        "schema": "rebotarm.joint_friction_model/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "joint": args.joint,
        "method": "steady bidirectional sweeps; tau minus nominal URDF gravity",
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "training_data": str(train),
        "training_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "validation_data": str(validation),
        "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "training_samples": len(y_train),
        "validation_samples": len(y_validation),
        "training_coefficients": {
            "viscous_nm_per_rad_s": float(train_coefficient[0]),
            "coulomb_nm": float(train_coefficient[1]),
            "offset_nm": float(train_coefficient[2]),
        },
        "validation_refit_coefficients": {
            "viscous_nm_per_rad_s": float(validation_coefficient[0]),
            "coulomb_nm": float(validation_coefficient[1]),
            "offset_nm": float(validation_coefficient[2]),
        },
        "deployed_coefficients": {
            "viscous_nm_per_rad_s": float(deployed[0]),
            "coulomb_nm": float(deployed[1]),
            "offset_nm": float(deployed[2]),
        },
        "viscous_repeatable": bool(fv_repeatable),
        "coulomb_relative_difference": fs_relative_difference,
        "offset_difference_nm": offset_difference,
        "training_rmse_nm": train_rmse,
        "validation_refit_rmse_nm": validation_refit_rmse,
        "validation_prediction_rmse_nm": validation_rmse,
        "training_design_condition_number": float(np.linalg.cond(X_train)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("joint friction calibration:", "PASS" if accepted else "FAIL")
    print("  train coefficients [fv, fs, off]:", train_coefficient)
    print("  validation refit:                ", validation_coefficient)
    print("  deployed:                        ", deployed)
    print(f"  validation prediction RMSE: {validation_rmse:.4f} N.m")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
