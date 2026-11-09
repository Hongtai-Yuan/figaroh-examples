#!/usr/bin/env python3
"""Design an offline, bounded multisine trajectory for reBotArm SysID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline
from scipy.linalg import qr


CENTER = np.array([0.0, 0.5235987756, 0.7853981634, -0.2508652382, 0.0, 0.0])
# Narrower than the URDF/driver hard limits and covered by prior forward poses.
Q_LOWER = np.array([-0.60, 0.18, 0.32, -0.70, -0.90, -0.90])
Q_UPPER = np.array([0.60, 0.92, 1.25, 0.20, 0.90, 0.90])
TARGET_AMPLITUDE = np.array([0.48, 0.27, 0.34, 0.30, 0.65, 0.65])
DQ_LIMIT = np.array([0.65, 0.65, 0.65, 0.85, 0.95, 1.10])
DDQ_LIMIT = np.array([1.8, 1.8, 1.8, 2.5, 3.0, 3.5])
JERK_LIMIT = np.array([12.0, 12.0, 12.0, 18.0, 22.0, 26.0])
EFFORT_LIMIT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
NOMINAL_TORQUE_LIMIT = 0.70 * EFFORT_LIMIT
FREQUENCIES_HZ = np.array([0.07, 0.11, 0.17, 0.23, 0.31])


def derivatives(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dq = np.gradient(q, dt, axis=0, edge_order=2)
    ddq = np.gradient(dq, dt, axis=0, edge_order=2)
    jerk = np.gradient(ddq, dt, axis=0, edge_order=2)
    return dq, ddq, jerk


def candidate_trajectory(rng, duration, rate, intensity):
    """Return a smooth, centre-to-centre random-phase multisine."""
    dt = 1.0 / rate
    t = np.arange(0.0, duration + 0.5 * dt, dt)
    hold = min(2.0, 0.08 * duration)
    active_duration = duration - 2.0 * hold
    active_t = np.clip(t - hold, 0.0, active_duration)
    active = (t >= hold) & (t <= duration - hold)
    window = np.zeros_like(t)
    window[active] = np.sin(np.pi * active_t[active] / active_duration) ** 2
    q = np.tile(CENTER, (len(t), 1))
    for joint in range(6):
        phases = rng.uniform(-np.pi, np.pi, len(FREQUENCIES_HZ))
        weights = rng.uniform(0.35, 1.0, len(FREQUENCIES_HZ))
        weights /= np.linalg.norm(weights)
        raw = sum(
            weight * np.sin(2.0 * np.pi * frequency * active_t + phase)
            for weight, frequency, phase in zip(weights, FREQUENCIES_HZ, phases)
        )
        shaped = window * raw
        peak = float(np.max(np.abs(shaped)))
        if peak < 1e-9:
            raise RuntimeError("degenerate multisine candidate")
        q[:, joint] += (
            intensity * TARGET_AMPLITUDE[joint] * rng.uniform(0.82, 1.0) * shaped / peak
        )
    q[t <= hold] = CENTER
    q[t >= duration - hold] = CENTER
    dq, ddq, jerk = derivatives(q, dt)
    return t, q, dq, ddq, jerk


def nominal_torque(model, data, q, dq, ddq):
    return np.asarray(
        [pin.rnea(model, data, qi, vi, ai).copy() for qi, vi, ai in zip(q, dq, ddq)]
    )


def report(q, dq, ddq, jerk, tau):
    return {
        "q_min_rad": q.min(0).tolist(),
        "q_max_rad": q.max(0).tolist(),
        "max_abs_velocity_rad_s": np.max(np.abs(dq), 0).tolist(),
        "max_abs_acceleration_rad_s2": np.max(np.abs(ddq), 0).tolist(),
        "max_abs_jerk_rad_s3": np.max(np.abs(jerk), 0).tolist(),
        "max_abs_nominal_torque_nm": np.max(np.abs(tau), 0).tolist(),
    }


def constraints_pass(q, dq, ddq, jerk, tau):
    return bool(
        np.all(q >= Q_LOWER)
        and np.all(q <= Q_UPPER)
        and np.all(np.max(np.abs(dq), 0) <= DQ_LIMIT)
        and np.all(np.max(np.abs(ddq), 0) <= DDQ_LIMIT)
        and np.all(np.max(np.abs(jerk), 0) <= JERK_LIMIT)
        and np.all(np.max(np.abs(tau), 0) <= NOMINAL_TORQUE_LIMIT)
    )


def spline_runtime_report(t, q):
    """Evaluate the same natural cubic interpolation used by dm_controller."""
    dense_t = np.arange(t[0], t[-1] + 0.5 / 300.0, 1.0 / 300.0)
    spline = CubicSpline(t, q, axis=0, bc_type="natural")
    dense_q = spline(dense_t)
    dense_dq = spline(dense_t, 1)
    return {
        "q_min_rad": dense_q.min(0).tolist(),
        "q_max_rad": dense_q.max(0).tolist(),
        "max_abs_velocity_rad_s": np.max(np.abs(dense_dq), 0).tolist(),
        "passed": bool(
            np.all(dense_q >= Q_LOWER)
            and np.all(dense_q <= Q_UPPER)
            and np.all(np.max(np.abs(dense_dq), 0) <= DQ_LIMIT)
        ),
    }


def augmented_regressor(model, data, q, dq, ddq):
    """Stack inertial, viscous, smooth-Coulomb and constant-bias columns."""
    blocks = []
    for qi, vi, ai in zip(q, dq, ddq):
        inertial = pin.computeJointTorqueRegressor(model, data, qi, vi, ai).copy()
        extra = np.zeros((6, 18))
        for joint in range(6):
            extra[joint, joint] = vi[joint]
            extra[joint, 6 + joint] = np.tanh(vi[joint] / 0.03)
            extra[joint, 12 + joint] = 1.0
        blocks.append(np.hstack((inertial, extra)))
    W = np.vstack(blocks)
    W /= np.tile(EFFORT_LIMIT, len(q))[:, None]
    return W


def observability_metrics(W):
    rms = np.sqrt(np.mean(W * W, axis=0))
    observable = rms > 1e-8
    scaled = W[:, observable] / rms[observable]
    _, R, pivots = qr(scaled, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(R))
    if not diagonal.size or diagonal[0] == 0.0:
        return {"rank": 0, "condition_number": float("inf"), "log_information": -float("inf")}
    rank = int(np.count_nonzero(diagonal > diagonal[0] * 1e-7))
    selected = scaled[:, pivots[:rank]]
    singular = np.linalg.svd(selected, compute_uv=False)
    return {
        "rank": rank,
        "condition_number": float(singular[0] / singular[-1]),
        "log_information": float(np.sum(np.log(np.maximum(singular, 1e-15)))),
        "nonzero_columns": int(np.count_nonzero(observable)),
        "selected_columns": np.flatnonzero(observable)[pivots[:rank]].tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--objective-rate", type=float, default=10.0)
    parser.add_argument("--candidates", type=int, default=80)
    parser.add_argument("--intensity", type=float, default=0.72)
    args = parser.parse_args()
    if not 30.0 <= args.duration <= 120.0:
        parser.error("duration must be in [30,120] seconds")
    if not 25.0 <= args.sample_rate <= 100.0:
        parser.error("sample-rate must be in [25,100] Hz")
    if not 5.0 <= args.objective_rate <= args.sample_rate:
        parser.error("objective-rate must be in [5,sample-rate] Hz")
    if args.candidates < 10 or not 0.2 <= args.intensity <= 1.0:
        parser.error("candidates >=10 and intensity in [0.2,1.0] required")
    return args


def main():
    args = parse_args()
    urdf = Path(args.urdf).resolve()
    output = Path(args.output).resolve()
    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 6 or model.nv != 6:
        raise ValueError(f"expected 6 DOF, got nq={model.nq}, nv={model.nv}")
    data = model.createData()
    rng = np.random.default_rng(args.seed)
    stride = max(1, int(round(args.sample_rate / args.objective_rate)))
    best = None
    feasible = rejected = 0
    for index in range(args.candidates):
        t, q, dq, ddq, jerk = candidate_trajectory(
            rng, args.duration, args.sample_rate, args.intensity
        )
        tau = nominal_torque(model, data, q, dq, ddq)
        spline_report = spline_runtime_report(t, q)
        if not constraints_pass(q, dq, ddq, jerk, tau) or not spline_report["passed"]:
            rejected += 1
            continue
        feasible += 1
        W = augmented_regressor(model, data, q[::stride], dq[::stride], ddq[::stride])
        metrics = observability_metrics(W)
        key = (metrics["rank"], -metrics["condition_number"], metrics["log_information"])
        if best is None or key > best[0]:
            best = (key, index, t, q, dq, ddq, jerk, tau, metrics, spline_report)
    if best is None:
        raise RuntimeError(f"no feasible candidate among {args.candidates}; reduce intensity")

    _, candidate_index, t, q, dq, ddq, jerk, tau, metrics, spline_report = best
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", *[f"q{i}" for i in range(1, 7)],
                         *[f"dq{i}" for i in range(1, 7)],
                         *[f"ddq{i}" for i in range(1, 7)]])
        for k in range(len(t)):
            writer.writerow([f"{t[k]:.9f}", *q[k], *dq[k], *ddq[k]])
    metadata = {
        "schema": "rebotarm.safe_sysid_trajectory/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "trajectory_csv": str(output),
        "trajectory_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "seed": args.seed,
        "candidate_index": candidate_index,
        "candidates_requested": args.candidates,
        "candidates_feasible": feasible,
        "candidates_rejected_by_limits": rejected,
        "duration_s": args.duration,
        "sample_rate_hz": args.sample_rate,
        "objective_rate_hz": args.objective_rate,
        "intensity": args.intensity,
        "centre_rad": CENTER.tolist(),
        "limits": {
            "position_lower_rad": Q_LOWER.tolist(),
            "position_upper_rad": Q_UPPER.tolist(),
            "velocity_rad_s": DQ_LIMIT.tolist(),
            "acceleration_rad_s2": DDQ_LIMIT.tolist(),
            "jerk_rad_s3": JERK_LIMIT.tolist(),
            "nominal_torque_nm": NOMINAL_TORQUE_LIMIT.tolist(),
        },
        "constraint_report": report(q, dq, ddq, jerk, tau),
        "driver_spline_300hz_report": spline_report,
        "observability": metrics,
        "hardware_executed": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"candidate={candidate_index}, feasible={feasible}, rejected={rejected}")
    print(f"rank={metrics['rank']}, scaled_condition={metrics['condition_number']:.3f}")
    print("max|dq|", np.array2string(np.max(np.abs(dq), 0), precision=3))
    print("max|ddq|", np.array2string(np.max(np.abs(ddq), 0), precision=3))
    print("max|tau_nom|", np.array2string(np.max(np.abs(tau), 0), precision=3))
    print("wrote", output)


if __name__ == "__main__":
    main()
