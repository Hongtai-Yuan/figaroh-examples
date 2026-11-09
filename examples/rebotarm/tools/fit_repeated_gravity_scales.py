#!/usr/bin/env python3
"""Fit joint gravity scales across several training and validation campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin


RMSE_LIMIT = np.array([0.20, 0.50, 0.50, 0.15, 0.15, 0.15])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, model: pin.Model, offset: np.ndarray):
    frame = pd.read_csv(path)
    data = model.createData()
    gravity, effort = [], []
    for _, group in frame.groupby("pose_id", sort=True):
        if set(group["approach"].astype(int)) != {-1, 1}:
            raise ValueError(f"{path}: every pose must contain both approaches")
        q = group[[f"q{i}" for i in range(1, 7)]].to_numpy(float).mean(axis=0)
        tau = group[[f"tau{i}" for i in range(1, 7)]].to_numpy(float).mean(axis=0)
        gravity.append(pin.computeGeneralizedGravity(model, data, q).copy())
        effort.append(tau - offset)
    return np.asarray(gravity), np.asarray(effort)


def rmse(prediction: np.ndarray, target: np.ndarray):
    return np.sqrt(np.mean((prediction - target) ** 2, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--validation", action="append", required=True)
    parser.add_argument("--friction-model", action="append", required=True)
    parser.add_argument("--output", default="results/static_gravity_model.json")
    args = parser.parse_args()
    if len(args.train) < 2 or len(args.validation) < 2 or len(args.friction_model) != 6:
        parser.error("need >=2 train, >=2 validation, and exactly 6 friction models")

    urdf = Path(args.urdf).resolve()
    trains = [Path(x).resolve() for x in args.train]
    validations = [Path(x).resolve() for x in args.validation]
    friction_paths = [Path(x).resolve() for x in args.friction_model]
    output = Path(args.output).resolve()
    offsets = []
    for joint, path in enumerate(friction_paths, 1):
        item = json.loads(path.read_text())
        if not item.get("accepted") or item.get("joint") != joint:
            raise ValueError(f"{path}: expected accepted joint {joint} model")
        offsets.append(item["deployed_coefficients"]["offset_nm"])
    offsets = np.asarray(offsets)
    model = pin.buildModelFromUrdf(str(urdf))
    train_sets = [load(path, model, offsets) for path in trains]
    validation_sets = [load(path, model, offsets) for path in validations]
    G = np.vstack([item[0] for item in train_sets])
    Y = np.vstack([item[1] for item in train_sets])
    denominator = np.sum(G * G, axis=0)
    fitted = np.ones(6)
    observable = denominator > 1e-8
    fitted[observable] = np.sum(G[:, observable] * Y[:, observable], axis=0) / denominator[observable]
    refits = []
    for gravity, effort in train_sets:
        den = np.sum(gravity * gravity, axis=0)
        value = np.ones(6)
        mask = den > 1e-8
        value[mask] = np.sum(gravity[:, mask] * effort[:, mask], axis=0) / den[mask]
        refits.append(value)
    refits = np.asarray(refits)

    VG = np.vstack([item[0] for item in validation_sets])
    VY = np.vstack([item[1] for item in validation_sets])
    nominal_rmse = rmse(VG, VY)
    fitted_rmse = rmse(VG * fitted, VY)
    improvement = 1.0 - fitted_rmse / np.maximum(nominal_rmse, 1e-12)
    pooled_range = np.ptp(VG, axis=0)
    selected = np.ones(6)
    diagnostics = []
    for joint in range(6):
        per_set_nonworse = []
        for gravity, effort in validation_sets:
            before = float(rmse(gravity, effort)[joint])
            after = float(rmse(gravity * fitted, effort)[joint])
            per_set_nonworse.append(after <= before + 1e-9)
        repeatability = float(
            np.max(np.abs(refits[:, joint] - fitted[joint]))
            / max(abs(fitted[joint]), 1e-12)
        )
        accept_joint = bool(
            pooled_range[joint] >= 0.20
            and improvement[joint] >= 0.20
            and all(per_set_nonworse)
            and repeatability <= 0.15
            and 0.50 <= fitted[joint] <= 1.50
        )
        if accept_joint:
            selected[joint] = fitted[joint]
        diagnostics.append(
            {
                "joint": joint + 1,
                "fitted_scale": float(fitted[joint]),
                "selected": accept_joint,
                "training_refit_scales": refits[:, joint].tolist(),
                "max_training_relative_difference": repeatability,
                "pooled_validation_gravity_range_nm": float(pooled_range[joint]),
                "pooled_nominal_rmse_nm": float(nominal_rmse[joint]),
                "pooled_scaled_rmse_nm": float(fitted_rmse[joint]),
                "pooled_improvement_fraction": float(improvement[joint]),
                "all_validation_sets_nonworse": all(per_set_nonworse),
            }
        )
    selected_rmse = rmse(VG * selected, VY)
    overall_nominal = float(np.sqrt(np.mean((VG - VY) ** 2)))
    overall_selected = float(np.sqrt(np.mean((VG * selected - VY) ** 2)))
    accepted = bool(
        np.any(selected != 1.0)
        and np.all(selected_rmse <= RMSE_LIMIT)
        and overall_selected < overall_nominal
    )
    calibrated_q = np.vstack(
        [
            pd.read_csv(path)[[f"q{i}" for i in range(1, 7)]].to_numpy(float)
            for path in [*trains, *validations]
        ]
    )
    result = {
        "schema": "rebotarm.static_gravity_model/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "deployment_status": "independently_validated_not_hardware_retested",
        "method": "paired-direction medians across repeated campaigns; known offsets removed",
        "urdf": str(urdf),
        "urdf_sha256": digest(urdf),
        "training_data": [{"file": str(p), "sha256": digest(p)} for p in trains],
        "validation_data": [{"file": str(p), "sha256": digest(p)} for p in validations],
        "friction_models": [{"file": str(p), "sha256": digest(p)} for p in friction_paths],
        "torque_offsets_subtracted_nm": offsets.tolist(),
        "gravity_scale": selected.tolist(),
        "unselected_fitted_scale": fitted.tolist(),
        "validation_rmse_nm": selected_rmse.tolist(),
        "validation_rmse_limit_nm": RMSE_LIMIT.tolist(),
        "validation_overall_nominal_rmse_nm": overall_nominal,
        "validation_overall_selected_rmse_nm": overall_selected,
        "calibrated_q_min_rad": calibrated_q.min(axis=0).tolist(),
        "calibrated_q_max_rad": calibrated_q.max(axis=0).tolist(),
        "joint_diagnostics": diagnostics,
        "limitations": [
            "joint1 and joint6 gravity are structurally unobservable in this mechanism",
            "standard URDF has no per-joint gravity-scale field; use this calibration in the gravity controller",
            "new scale requires a short low-gain hardware hold test before manual-drag deployment",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("repeated gravity calibration:", "PASS" if accepted else "FAIL")
    print("selected", selected)
    print("pooled nominal", nominal_rmse)
    print("pooled selected", selected_rmse)
    print("overall", overall_nominal, "->", overall_selected)
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
