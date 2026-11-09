#!/usr/bin/env python3
"""Fit friction around the accepted gravity model and verify on a held-out run.

The rigid-body URDF remains unchanged.  Hyperparameters are selected only from
blocked cross-validation of the training recording; the validation recording is
opened afterwards and is used once for the declared acceptance decision.
"""

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
from scipy.optimize import lsq_linear


JOINTS = 6
WIDTH_GRID = (0.005, 0.01, 0.02, 0.03, 0.05)
RIDGE_GRID = (10.0, 31.6227766, 100.0, 316.227766, 1000.0, 3162.27766)
RMSE_LIMIT_NM = np.array([0.30, 0.40, 0.81, 0.15, 0.15, 0.15])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = ["t", *[f"q{i}" for i in range(1, 7)], *[f"tau{i}" for i in range(1, 7)]]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    t = frame["t"].to_numpy(float)
    q = frame[[f"q{i}" for i in range(1, 7)]].to_numpy(float)
    tau = frame[[f"tau{i}" for i in range(1, 7)]].to_numpy(float)
    if len(t) < 1000 or np.any(~np.isfinite(q)) or np.any(~np.isfinite(tau)):
        raise ValueError(f"{path}: insufficient or non-finite data")
    dt = np.diff(t)
    if np.any(dt <= 0) or abs(float(np.median(dt)) - 0.01) > 1e-5:
        raise ValueError(f"{path}: expected strictly increasing 100 Hz timestamps")
    return t, q, tau


def nominal_and_velocity(
    model: pin.Model,
    q_raw: np.ndarray,
    gravity_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = model.createData()
    q = sosfiltfilt(butter(4, 8.0, fs=100.0, output="sos"), q_raw, axis=0)
    velocity = np.gradient(q, 0.01, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, 0.01, axis=0, edge_order=2)
    nominal = []
    for qi, vi, ai in zip(q, velocity, acceleration):
        tau = pin.rnea(model, data, qi, vi, ai).copy()
        gravity = pin.computeGeneralizedGravity(model, data, qi).copy()
        nominal.append(tau + (gravity_scale - 1.0) * gravity)
    return np.asarray(nominal), velocity


def friction_regressor(velocity: np.ndarray, width: float) -> np.ndarray:
    rows = []
    for vi in velocity:
        block = np.zeros((JOINTS, 3 * JOINTS))
        for joint in range(JOINTS):
            block[joint, joint] = vi[joint]
            block[joint, JOINTS + joint] = np.tanh(vi[joint] / width)
            block[joint, 2 * JOINTS + joint] = 1.0
        rows.append(block)
    return np.asarray(rows)


def solve_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flat_x = x.reshape(-1, 3 * JOINTS)
    flat_y = y.reshape(-1)
    if row_mask is None:
        row_mask = np.ones(len(flat_y), dtype=bool)
    scale = np.sqrt(np.mean(flat_x[row_mask] ** 2, axis=0))
    scale[scale < 1e-12] = 1.0
    scaled = flat_x / scale
    augmented_x = np.vstack((scaled[row_mask], np.sqrt(ridge) * np.eye(3 * JOINTS)))
    augmented_y = np.concatenate((flat_y[row_mask], np.zeros(3 * JOINTS)))
    # Viscous and Coulomb friction dissipate energy, hence their coefficients
    # cannot be negative.  Constant sensor/model biases remain unconstrained.
    lower = np.r_[np.zeros(2 * JOINTS), np.full(JOINTS, -np.inf)]
    upper = np.full(3 * JOINTS, np.inf)
    solution = lsq_linear(
        augmented_x, augmented_y, bounds=(lower * scale, upper * scale),
        method="bvls", tol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"constrained ridge solve failed: {solution.message}")
    coefficient = solution.x / scale
    return coefficient, flat_x @ coefficient


def rmse(error: np.ndarray) -> tuple[float, np.ndarray]:
    return float(np.sqrt(np.mean(error**2))), np.sqrt(np.mean(error**2, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--gravity-model", default="results/static_gravity_model.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="results/validated_residual_dynamics_v2.json")
    args = parser.parse_args()

    urdf = Path(args.urdf).resolve()
    gravity_path = Path(args.gravity_model).resolve()
    train_path = Path(args.train).resolve()
    validation_path = Path(args.validation).resolve()
    output = Path(args.output).resolve()
    if train_path == validation_path or sha256(train_path) == sha256(validation_path):
        raise ValueError("training and validation recordings must be independent")

    gravity_model = json.loads(gravity_path.read_text())
    if gravity_model.get("schema") != "rebotarm.static_gravity_model/v1" or not gravity_model.get("accepted"):
        raise ValueError("gravity calibration is not an accepted static_gravity_model/v1")
    if gravity_model.get("urdf_sha256") != sha256(urdf):
        raise ValueError("gravity calibration and URDF hashes do not match")
    gravity_scale = np.asarray(gravity_model["gravity_scale"], dtype=float)
    if gravity_scale.shape != (JOINTS,) or np.any(~np.isfinite(gravity_scale)):
        raise ValueError("gravity_scale must have six finite entries")

    model = pin.buildModelFromUrdf(str(urdf))
    _, q_train, tau_train = load_csv(train_path)
    nominal_train, velocity_train = nominal_and_velocity(model, q_train, gravity_scale)
    residual_train = tau_train - nominal_train

    # Alternating five-second blocks prevent adjacent samples from appearing on
    # opposite sides of the internal split.  No validation data is read here.
    block = (np.arange(len(q_train)) // 500) % 2
    fit_rows = np.repeat(block == 0, JOINTS)
    score_rows = np.repeat(block == 1, JOINTS)
    trials = []
    best = None
    for width in WIDTH_GRID:
        design = friction_regressor(velocity_train, width)
        for ridge in RIDGE_GRID:
            coefficient, prediction = solve_ridge(design, residual_train, ridge, fit_rows)
            error = prediction[score_rows] - residual_train.reshape(-1)[score_rows]
            score = float(np.sqrt(np.mean(error**2)))
            trials.append({"transition_width_rad_s": width, "ridge": ridge, "rmse_nm": score})
            if best is None or score < best[0]:
                best = (score, width, ridge)

    cv_rmse, width, ridge = best
    design_train = friction_regressor(velocity_train, width)
    coefficient, train_flat_prediction = solve_ridge(design_train, residual_train, ridge)
    train_prediction = nominal_train + train_flat_prediction.reshape(-1, JOINTS)
    train_overall, train_joint = rmse(train_prediction - tau_train)

    # The held-out file is deliberately loaded only after all model choices.
    _, q_validation, tau_validation = load_csv(validation_path)
    nominal_validation, velocity_validation = nominal_and_velocity(
        model, q_validation, gravity_scale
    )
    design_validation = friction_regressor(velocity_validation, width)
    friction_validation = design_validation.reshape(-1, 3 * JOINTS) @ coefficient
    prediction_validation = nominal_validation + friction_validation.reshape(-1, JOINTS)
    validation_overall, validation_joint = rmse(prediction_validation - tau_validation)
    nominal_overall, nominal_joint = rmse(nominal_validation - tau_validation)
    improvement = 1.0 - validation_overall / nominal_overall

    coefficient_groups = {
        "viscous_nm_per_rad_s": coefficient[:JOINTS].tolist(),
        "coulomb_nm": coefficient[JOINTS : 2 * JOINTS].tolist(),
        "constant_bias_nm": coefficient[2 * JOINTS :].tolist(),
    }
    checks = {
        "independent_files": True,
        "validation_joint_rmse_within_limit": (validation_joint <= RMSE_LIMIT_NM).tolist(),
        "validation_overall_improvement_at_least_20_percent": bool(improvement >= 0.20),
        "finite_coefficients": bool(np.all(np.isfinite(coefficient))),
    }
    accepted = bool(
        all(checks["validation_joint_rmse_within_limit"])
        and checks["validation_overall_improvement_at_least_20_percent"]
        and checks["finite_coefficients"]
    )
    result = {
        "schema": "rebotarm.validated_residual_dynamics/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "deployment_status": "validated_not_hardware_deployed" if accepted else "rejected",
        "method": "accepted static gravity plus regularized viscous/Coulomb/bias residual",
        "model_scope": "dynamic torque prediction; rigid inertial parameters remain nominal",
        "urdf": str(urdf),
        "urdf_sha256": sha256(urdf),
        "gravity_model": str(gravity_path),
        "gravity_model_sha256": sha256(gravity_path),
        "training_data": str(train_path),
        "training_sha256": sha256(train_path),
        "validation_data": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "hyperparameter_selection": {
            "policy": "alternating 5 s blocked split of training data only",
            "selected_transition_width_rad_s": width,
            "selected_ridge": ridge,
            "internal_cv_rmse_nm": cv_rmse,
            "trials": trials,
        },
        "gravity_scale": gravity_scale.tolist(),
        "coefficients": coefficient_groups,
        "training_rmse_nm": {"overall": train_overall, "per_joint": train_joint.tolist()},
        "validation": {
            "nominal_rmse_nm": {"overall": nominal_overall, "per_joint": nominal_joint.tolist()},
            "identified_rmse_nm": {"overall": validation_overall, "per_joint": validation_joint.tolist()},
            "overall_improvement_fraction": improvement,
            "per_joint_limit_nm": RMSE_LIMIT_NM.tolist(),
        },
        "acceptance_checks": checks,
        "limitations": [
            "absolute motor torque scale is not traceable to a calibrated external load cell",
            "validated only inside the recorded position, velocity and payload envelope",
            "constant bias may contain unmodelled gravity error and is not enabled for manual drag",
            "hardware deployment requires a separate low-gain safety test",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("validated residual dynamics:", "PASS" if accepted else "FAIL")
    print(f"selected width={width:g} rad/s ridge={ridge:g}; internal CV RMSE={cv_rmse:.4f} Nm")
    print(f"validation nominal={nominal_overall:.4f} identified={validation_overall:.4f} Nm")
    print(f"improvement={100 * improvement:.1f}%")
    print("per-joint RMSE", np.array2string(validation_joint, precision=4))
    print("wrote", output)
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
