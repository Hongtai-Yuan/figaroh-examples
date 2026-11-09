#!/usr/bin/env python3
"""Fit a regularized gravity-only mass/COM URDF with two held-out checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin


RIDGE_GRID = np.logspace(-3, 3, 13)
RMSE_LIMIT = np.array([0.10, 0.20, 0.35, 0.15, 0.10, 0.06])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_paired(path: Path, model: pin.Model, columns: list[int], offset: np.ndarray):
    frame = pd.read_csv(path)
    blocks, efforts = [], []
    data = model.createData()
    for _, group in frame.groupby("pose_id", sort=True):
        approaches = set(group["approach"].astype(int))
        if approaches != {-1, 1}:
            raise ValueError(f"{path}: each pose needs approach -1 and +1")
        q = group[[f"q{i}" for i in range(1, 7)]].to_numpy(float).mean(axis=0)
        tau = group[[f"tau{i}" for i in range(1, 7)]].to_numpy(float).mean(axis=0)
        blocks.append(
            pin.computeJointTorqueRegressor(
                model, data, q, np.zeros(6), np.zeros(6)
            )[:, columns].copy()
        )
        efforts.append(tau - offset)
    return np.asarray(blocks), np.asarray(efforts)


def solve(A: np.ndarray, y: np.ndarray, nominal: np.ndarray, scale: np.ndarray, ridge: float):
    X = A.reshape(-1, len(nominal))
    target = y.reshape(-1)
    Z = X * scale
    delta = np.linalg.solve(
        Z.T @ Z + ridge * np.eye(len(nominal)), Z.T @ (target - X @ nominal)
    )
    return nominal + scale * delta


def metrics(A: np.ndarray, y: np.ndarray, parameters: np.ndarray):
    error = np.einsum("nij,j->ni", A, parameters) - y
    return {
        "overall_rmse_nm": float(np.sqrt(np.mean(error**2))),
        "per_joint_rmse_nm": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--friction-model", action="append", required=True)
    parser.add_argument("--output-urdf", default="urdf/rebotarm_gravity_identified.urdf")
    parser.add_argument("--output-result", default="results/gravity_mass_com_v3.json")
    args = parser.parse_args()
    if len(args.friction_model) != 6:
        parser.error("exactly six --friction-model arguments are required")

    urdf = Path(args.urdf).resolve()
    train, validation, repeat = [Path(x).resolve() for x in (args.train, args.validation, args.repeat)]
    output_urdf = Path(args.output_urdf).resolve()
    output_result = Path(args.output_result).resolve()
    model_paths = [Path(x).resolve() for x in args.friction_model]
    offsets = []
    for joint, path in enumerate(model_paths, 1):
        item = json.loads(path.read_text())
        if not item.get("accepted") or item.get("joint") != joint:
            raise ValueError(f"{path}: expected accepted joint {joint} model")
        offsets.append(item["deployed_coefficients"]["offset_nm"])
    offsets = np.asarray(offsets, dtype=float)

    model = pin.buildModelFromUrdf(str(urdf))
    columns = [10 * link + term for link in range(6) for term in range(4)]
    full_nominal = np.concatenate(
        [model.inertias[joint].toDynamicParameters() for joint in range(1, 7)]
    )
    nominal = full_nominal[columns]
    scale = np.maximum(np.abs(nominal), 0.05)
    A, y = load_paired(train, model, columns, offsets)
    V, z = load_paired(validation, model, columns, offsets)
    R, r = load_paired(repeat, model, columns, offsets)

    cv = []
    for ridge in RIDGE_GRID:
        fold_rmse = []
        for held_out in range(len(A)):
            keep = np.arange(len(A)) != held_out
            parameters = solve(A[keep], y[keep], nominal, scale, float(ridge))
            error = A[held_out] @ parameters - y[held_out]
            fold_rmse.append(float(np.sqrt(np.mean(error**2))))
        cv.append({"ridge": float(ridge), "mean_rmse_nm": float(np.mean(fold_rmse))})
    selected_ridge = min(cv, key=lambda item: item["mean_rmse_nm"])["ridge"]
    parameters = solve(A, y, nominal, scale, selected_ridge)

    masses, centres = [], []
    for link in range(6):
        mass = float(parameters[4 * link])
        centre = parameters[4 * link + 1 : 4 * link + 4] / mass
        masses.append(mass)
        centres.append(centre.tolist())
    masses_array, centres_array = np.asarray(masses), np.asarray(centres)
    train_metrics = metrics(A, y, parameters)
    validation_metrics = metrics(V, z, parameters)
    repeat_metrics = metrics(R, r, parameters)
    nominal_validation = metrics(V, z, nominal)
    nominal_repeat = metrics(R, r, nominal)
    checks = {
        "positive_mass": bool(np.all(masses_array > 0.0)),
        "com_within_0_30_m": bool(np.max(np.abs(centres_array)) <= 0.30),
        "validation_per_joint_rmse": bool(
            np.all(np.asarray(validation_metrics["per_joint_rmse_nm"]) <= RMSE_LIMIT)
        ),
        "repeat_per_joint_rmse": bool(
            np.all(np.asarray(repeat_metrics["per_joint_rmse_nm"]) <= RMSE_LIMIT)
        ),
        "validation_improvement_20_percent": bool(
            validation_metrics["overall_rmse_nm"]
            <= 0.8 * nominal_validation["overall_rmse_nm"]
        ),
        "repeat_improvement_20_percent": bool(
            repeat_metrics["overall_rmse_nm"] <= 0.8 * nominal_repeat["overall_rmse_nm"]
        ),
    }
    accepted = all(checks.values())

    if accepted:
        parser_xml = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(urdf, parser=parser_xml)
        root = tree.getroot()
        for link_index in range(6):
            link = root.find(f"./link[@name='link{link_index + 1}']")
            if link is None or link.find("inertial") is None:
                raise ValueError(f"link{link_index + 1} inertial element missing")
            inertial = link.find("inertial")
            inertial.find("mass").set("value", f"{masses[link_index]:.15g}")
            inertial.find("origin").set(
                "xyz", " ".join(f"{value:.15g}" for value in centres[link_index])
            )
        output_urdf.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
        exported = pin.buildModelFromUrdf(str(output_urdf))
        for link_index in range(6):
            if not np.isclose(exported.inertias[link_index + 1].mass, masses[link_index]):
                raise RuntimeError("exported URDF mass round-trip failed")

    result = {
        "schema": "rebotarm.gravity_mass_com/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "deployment_scope": "gravity_only",
        "urdf_source": str(urdf),
        "urdf_source_sha256": sha256(urdf),
        "urdf_output": str(output_urdf) if accepted else None,
        "urdf_output_sha256": sha256(output_urdf) if accepted else None,
        "data": {
            "train": {"file": str(train), "sha256": sha256(train)},
            "validation": {"file": str(validation), "sha256": sha256(validation)},
            "repeat": {"file": str(repeat), "sha256": sha256(repeat)},
        },
        "friction_models": [
            {"file": str(path), "sha256": sha256(path)} for path in model_paths
        ],
        "torque_offsets_subtracted_nm": offsets.tolist(),
        "hyperparameter_policy": "leave-one-pose-out CV on training only",
        "ridge_trials": cv,
        "selected_ridge": selected_ridge,
        "identified_mass_kg": masses,
        "identified_com_m": centres,
        "rotational_inertia_policy": "unchanged from source URDF",
        "training": train_metrics,
        "validation": validation_metrics,
        "repeat": repeat_metrics,
        "nominal_validation": nominal_validation,
        "nominal_repeat": nominal_repeat,
        "per_joint_rmse_limit_nm": RMSE_LIMIT.tolist(),
        "acceptance_checks": checks,
        "limitations": [
            "mass and COM are regularized effective gravity parameters, not independently weighed CAD truth",
            "exported URDF is accepted for gravity calculation only",
            "rotational inertias remain unchanged because dynamic cross-pose validation failed",
        ],
    }
    output_result.parent.mkdir(parents=True, exist_ok=True)
    output_result.write_text(json.dumps(result, indent=2) + "\n")
    print("gravity mass/COM:", "PASS" if accepted else "FAIL")
    print("selected ridge", selected_ridge)
    print("validation", validation_metrics)
    print("repeat", repeat_metrics)
    print("masses", masses_array)
    print("COM", centres_array)
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
