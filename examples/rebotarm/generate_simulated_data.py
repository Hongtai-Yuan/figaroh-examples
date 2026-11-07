#!/usr/bin/env python3
"""Generate independent RNEA truth fixtures to self-test the identification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin

from figaroh.identification.identification_tools import (
    calculate_first_second_order_differentiation,
)
from figaroh.identification.config import unified_to_legacy_identif_config
from figaroh.tools.robot import load_robot
from figaroh.utils.config_parser import get_param_from_yaml


def trajectory(t: np.ndarray, seed: int) -> np.ndarray:
    """Persistent, smooth multi-sine motion well inside the URDF limits."""
    rng = np.random.default_rng(seed)
    center = np.array([0.0, 1.55, 1.72, 0.05, 0.0, 0.0])
    amplitude = np.array([0.75, 0.50, 0.55, 0.48, 0.48, 0.80])
    q = np.tile(center, (len(t), 1))
    # Different irrational-ish frequencies and phases reduce regressor correlation.
    for joint in range(6):
        weights = rng.uniform(0.25, 1.0, 3)
        weights /= np.sum(weights)
        frequencies = rng.uniform(0.055, 0.28, 3)
        phases = rng.uniform(-np.pi, np.pi, 3)
        for weight, frequency, phase in zip(weights, frequencies, phases):
            q[:, joint] += (
                amplitude[joint]
                * weight
                * np.sin(2 * np.pi * frequency * t + phase)
            )
    return q


def write_split(path: Path, robot, ident_config: dict, seed: int, duration: float) -> None:
    dt = float(ident_config["ts"])
    t = np.arange(0.0, duration, dt)
    q = trajectory(t, seed)
    q_kept, dq, ddq = calculate_first_second_order_differentiation(
        robot.model, q, ident_config
    )
    count = len(q_kept)
    data = robot.model.createData()
    tau = np.array(
        [pin.rnea(robot.model, data, q_kept[i], dq[i], ddq[i]) for i in range(count)]
    )
    # The loader consumes n-2 rows. Pad only to keep every CSV channel equal-length.
    tau_full = np.vstack([tau, np.repeat(tau[-1][None, :], len(t) - count, axis=0)])
    frame = pd.DataFrame({"t": t})
    for i in range(6):
        frame[f"q{i + 1}"] = q[:, i]
    for i in range(6):
        frame[f"tau{i + 1}"] = tau_full[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.identification_log/v1",
                "simulated": True,
                "mapping_verified": True,
                "torque_source": "Pinocchio RNEA using nominal URDF",
                "seed": seed,
                "sample_rate_hz": 1.0 / dt,
                "independent_split": path.stem,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path} ({len(frame)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--config", default="config/rebotarm_simulated.yaml")
    parser.add_argument("--output-dir", default="data/simulated")
    parser.add_argument("--duration", type=float, default=35.0)
    args = parser.parse_args()
    robot = load_robot(args.urdf, package_dirs="../../models", load_by_urdf=True)
    unified = get_param_from_yaml(robot, args.config, "identification")
    ident_config = unified_to_legacy_identif_config(robot, unified)
    output = Path(args.output_dir)
    write_split(output / "train.csv", robot, ident_config, seed=1701, duration=args.duration)
    write_split(
        output / "validation.csv", robot, ident_config, seed=2903, duration=args.duration
    )


if __name__ == "__main__":
    main()

