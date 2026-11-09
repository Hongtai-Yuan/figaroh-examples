#!/usr/bin/env python3
"""Generate a bounded bidirectional single-joint friction trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pinocchio as pin

from generate_safe_sysid_trajectory import (
    CENTER,
    DDQ_LIMIT,
    DQ_LIMIT,
    JERK_LIMIT,
    NOMINAL_TORQUE_LIMIT,
    Q_LOWER,
    Q_UPPER,
    constraints_pass,
    derivatives,
    nominal_torque,
    report,
    spline_runtime_report,
)


AMPLITUDE = np.array([0.35, 0.18, 0.22, 0.22, 0.42, 0.48])
SPEEDS = (
    (0.12, 0.28),
    (0.08, 0.18),
    (0.08, 0.18),
    (0.10, 0.24),
    (0.14, 0.34),
    (0.16, 0.40),
)


def append_hold(values: list[float], value: float, seconds: float, rate: float) -> None:
    values.extend([value] * max(1, int(round(seconds * rate))))


def append_minimum_jerk(
    values: list[float], start: float, end: float, seconds: float, rate: float
) -> None:
    count = max(2, int(round(seconds * rate)))
    u = np.arange(1, count + 1, dtype=float) / count
    shape = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    values.extend((start + (end - start) * shape).tolist())


def append_sweep(
    values: list[float], start: float, end: float, speed: float, rate: float
) -> None:
    distance = abs(end - start)
    direction = np.sign(end - start)
    ramp = min(0.8, distance / (2.0 * speed))
    constant = max(0.0, distance / speed - ramp)
    duration = 2.0 * ramp + constant
    count = max(4, int(round(duration * rate)))
    t = np.arange(1, count + 1, dtype=float) / rate
    velocity = np.empty(count)
    for index, stamp in enumerate(t):
        if stamp < ramp:
            velocity[index] = 0.5 * speed * (1.0 - np.cos(np.pi * stamp / ramp))
        elif stamp < ramp + constant:
            velocity[index] = speed
        else:
            u = min(ramp, stamp - ramp - constant)
            velocity[index] = 0.5 * speed * (1.0 + np.cos(np.pi * u / ramp))
    position = np.cumsum(velocity) / rate
    position *= distance / position[-1]
    values.extend((start + direction * position).tolist())


def make_trajectory(joint: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    centre = float(CENTER[joint])
    lower = centre - float(AMPLITUDE[joint])
    upper = centre + float(AMPLITUDE[joint])
    values: list[float] = []
    append_hold(values, centre, 2.0, rate)
    for speed in SPEEDS[joint]:
        append_minimum_jerk(values, centre, lower, 2.5, rate)
        append_hold(values, lower, 0.8, rate)
        append_sweep(values, lower, upper, speed, rate)
        append_hold(values, upper, 0.8, rate)
        append_minimum_jerk(values, upper, centre, 2.5, rate)
        append_hold(values, centre, 1.0, rate)
        append_minimum_jerk(values, centre, upper, 2.5, rate)
        append_hold(values, upper, 0.8, rate)
        append_sweep(values, upper, lower, speed, rate)
        append_hold(values, lower, 0.8, rate)
        append_minimum_jerk(values, lower, centre, 2.5, rate)
        append_hold(values, centre, 1.0, rate)
    q = np.tile(CENTER, (len(values) + 1, 1))
    q[1:, joint] = values
    t = np.arange(len(q), dtype=float) / rate
    return t, q


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    args = parser.parse_args()
    if not 25.0 <= args.sample_rate <= 100.0:
        parser.error("sample-rate must be in [25,100] Hz")
    return args


def main() -> None:
    args = parse_args()
    joint = args.joint - 1
    urdf = Path(args.urdf).resolve()
    output = Path(args.output).resolve()
    t, q = make_trajectory(joint, args.sample_rate)
    dq, ddq, jerk = derivatives(q, 1.0 / args.sample_rate)
    model = pin.buildModelFromUrdf(str(urdf))
    tau = nominal_torque(model, model.createData(), q, dq, ddq)
    spline_report = spline_runtime_report(t, q)
    if not constraints_pass(q, dq, ddq, jerk, tau) or not spline_report["passed"]:
        raise RuntimeError("generated friction trajectory violated a safety limit")
    if t[-1] > 120.0:
        raise RuntimeError(f"trajectory duration {t[-1]:.1f}s exceeds 120s")
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
        "trajectory_type": "single_joint_bidirectional_friction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "trajectory_csv": str(output),
        "trajectory_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "seed": None,
        "joint": args.joint,
        "commanded_speeds_rad_s": list(SPEEDS[joint]),
        "duration_s": float(t[-1]),
        "sample_rate_hz": args.sample_rate,
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
        "hardware_executed": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"J{args.joint}: duration={t[-1]:.1f}s speeds={SPEEDS[joint]}")
    print("max|dq|", np.array2string(np.max(np.abs(dq), 0), precision=3))
    print("max|ddq|", np.array2string(np.max(np.abs(ddq), 0), precision=3))
    print("wrote", output)


if __name__ == "__main__":
    main()
