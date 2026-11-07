#!/usr/bin/env python3
"""Fit conservative per-joint gravity scales from paired static measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pinocchio as pin


JOINTS = [f"joint{i}" for i in range(1, 7)]
RMSE_LIMIT = np.array([0.20, 0.50, 0.50, 0.15, 0.15, 0.15])


def load_paired(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = np.genfromtxt(path, delimiter=",", names=True)
    ids = np.asarray(frame["pose_id"], dtype=int)
    q, tau = [], []
    for pose_id in sorted(set(ids)):
        rows = ids == pose_id
        if np.count_nonzero(rows) != 2 or set(np.asarray(frame["approach"])[rows]) != {-1.0, 1.0}:
            raise ValueError(f"{path}: pose {pose_id} is not a paired approach measurement")
        q.append([np.mean(frame[f"q{i}"][rows]) for i in range(1, 7)])
        tau.append([np.mean(frame[f"tau{i}"][rows]) for i in range(1, 7)])
    return np.asarray(q), np.asarray(tau)


def gravity(model, data, q: np.ndarray) -> np.ndarray:
    return np.asarray([pin.computeGeneralizedGravity(model, data, row).copy() for row in q])


def rmse(prediction: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - measured) ** 2, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--train", default="data/real/static_targeted_precise.csv")
    parser.add_argument("--validation", default="data/real/static_validation_precise.csv")
    parser.add_argument("--output", default="results/static_gravity_model.json")
    args = parser.parse_args()

    urdf, train, validation = map(lambda x: Path(x).resolve(), (args.urdf, args.train, args.validation))
    q_train, tau_train = load_paired(train)
    q_val, tau_val = load_paired(validation)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    g_train, g_val = gravity(model, data, q_train), gravity(model, data, q_val)

    fitted_scale = np.ones(6)
    selected_scale = np.ones(6)
    nominal_rmse = rmse(g_val, tau_val)
    diagnostics = []
    for joint in range(6):
        denominator = float(g_train[:, joint] @ g_train[:, joint])
        scale = 1.0 if denominator < 1e-10 else float(
            (g_train[:, joint] @ tau_train[:, joint]) / denominator
        )
        fitted_scale[joint] = scale
        prediction = scale * g_val[:, joint]
        corrected_rmse = float(np.sqrt(np.mean((prediction - tau_val[:, joint]) ** 2)))
        improvement = 1.0 - corrected_rmse / max(float(nominal_rmse[joint]), 1e-12)
        variance = float(np.sum((tau_val[:, joint] - np.mean(tau_val[:, joint])) ** 2))
        r2 = 1.0 - float(np.sum((prediction - tau_val[:, joint]) ** 2)) / max(variance, 1e-12)
        gravity_range = float(np.ptp(g_val[:, joint]))
        selected = bool(
            gravity_range >= 0.20
            and improvement >= 0.20
            and r2 >= 0.50
            and 0.50 <= scale <= 1.50
        )
        if selected:
            selected_scale[joint] = scale
        diagnostics.append(
            {
                "joint": JOINTS[joint],
                "fitted_scale": scale,
                "selected": selected,
                "gravity_range_nm": gravity_range,
                "nominal_rmse_nm": float(nominal_rmse[joint]),
                "scaled_rmse_nm": corrected_rmse,
                "improvement_fraction": improvement,
                "validation_r2": r2,
            }
        )

    prediction = g_val * selected_scale
    selected_rmse = rmse(prediction, tau_val)
    overall_nominal = float(np.sqrt(np.mean((g_val - tau_val) ** 2)))
    overall_selected = float(np.sqrt(np.mean((prediction - tau_val) ** 2)))
    accepted = bool(
        np.all(selected_rmse <= RMSE_LIMIT)
        and overall_selected <= 0.15
        and overall_selected < overall_nominal
        and np.any(selected_scale != 1.0)
    )
    result = {
        "schema": "rebotarm.static_gravity_model/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "deployment_status": "offline_only",
        "method": "paired-direction static median; scale-only correction selected on independent validation",
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "training_data": str(train),
        "validation_data": str(validation),
        "gravity_scale": selected_scale.tolist(),
        "unselected_fitted_scale": fitted_scale.tolist(),
        "validation_rmse_nm": selected_rmse.tolist(),
        "validation_rmse_limit_nm": RMSE_LIMIT.tolist(),
        "validation_overall_nominal_rmse_nm": overall_nominal,
        "validation_overall_selected_rmse_nm": overall_selected,
        "calibrated_q_min_rad": np.minimum(q_train.min(0), q_val.min(0)).tolist(),
        "calibrated_q_max_rad": np.maximum(q_train.max(0), q_val.max(0)).tolist(),
        "joint_diagnostics": diagnostics,
        "limitations": [
            "no known calibration payload was attached, so absolute motor torque scale is not independently traceable",
            "model is validated for static gravity inside the recorded joint envelope only",
            "constant feedback offsets were not added to commanded torque",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("static gravity calibration:", "PASS" if accepted else "FAIL")
    print("  selected scale:", np.array2string(selected_scale, precision=5))
    print("  nominal RMSE: ", np.array2string(nominal_rmse, precision=4))
    print("  selected RMSE:", np.array2string(selected_rmse, precision=4))
    print(f"  overall: {overall_nominal:.4f} -> {overall_selected:.4f} N.m")
    print("  output:", output)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
