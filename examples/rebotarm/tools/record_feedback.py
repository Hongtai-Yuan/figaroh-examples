#!/usr/bin/env python3
"""Record raw motorbridge feedback without enabling or commanding the arm.

Run this with the reBotArm Python 3.10 environment, not ``figaroh-dev``.
This is useful for zero/direction checks and passive-motion logs. Whether the
drive's disabled-state torque estimate is meaningful must be verified before
using such a log for dynamics identification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--arm-config", default=None)
    args = parser.parse_args()
    if args.duration <= 0 or args.rate <= 0:
        parser.error("duration and rate must be positive")

    workspace = Path(__file__).resolve().parents[5]
    driver_root = workspace / "vendor" / "reBotArm_control_py"
    sys.path.insert(0, str(driver_root))
    from reBotArm_control_py.actuator import RobotArm

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arm = RobotArm(args.arm_config)
    rows = []
    period = 1.0 / args.rate
    started = time.monotonic()
    deadline = started
    try:
        while True:
            now = time.monotonic()
            if now - started >= args.duration:
                break
            arm.get_positions(request=True)
            q, dq, tau = arm.get_state()
            rows.append([now - started, *q.tolist(), *dq.tolist(), *tau.tolist()])
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        # RobotArm.disconnect disables before closing; this script never enables.
        arm.disconnect()

    names = ["t_raw"]
    names += [f"q_motor{i}" for i in range(1, 7)]
    names += [f"dq_motor{i}" for i in range(1, 7)]
    names += [f"tau_motor{i}" for i in range(1, 7)]
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.raw_feedback/v1",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "requested_rate_hz": args.rate,
                "rows": len(rows),
                "enabled_or_commanded_by_recorder": False,
                "warning": "Torque validity while disabled is not established.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(rows)} rows); no enable or motion command was sent")


if __name__ == "__main__":
    main()

