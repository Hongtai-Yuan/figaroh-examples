#!/usr/bin/env python3
"""Execute a protected identification trajectory and record joint position/torque."""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray


JOINTS = [f"joint{i}" for i in range(1, 7)]
CENTER = np.array([1.6707963268, 0.5235987756, 0.7853981634, -0.2508652382, 0.0, 0.0])
FORWARD_CENTER = CENTER.copy()
FORWARD_CENTER[0] = 0.0
AMPLITUDE = np.array([0.20, 0.16, 0.16, 0.14, 0.14, 0.18])
LOWER = np.array([-3.14, -1.57, -1.57, -2.20, -2.20, -2.20])
UPPER = np.array([3.14, 1.57, 1.57, 2.20, 2.20, 2.20])
EFFORT_LIMIT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
SEARCH_POSES = np.array(
    [
        [0.492699, 0.549779, 0.706858, -0.250865, -1.195551, 0.0],
        [1.670796, 0.130900, 0.280998, 0.141834, 0.043633, 0.043633],
        [1.670796, 0.916298, 1.289799, -0.643564, -0.043633, -0.043633],
        # The vision-search value is 2.849 rad; use 2.20 so the identification
        # data remains inside this URDF's joint1 limit (2.355 rad).
        [2.200000, 0.497419, 0.863938, -0.250865, 1.195551, 0.0],
    ]
)


class Recorder(Node):
    def __init__(self, broad: bool = False) -> None:
        super().__init__("rebotarm_identification_recorder")
        self.lock = threading.Lock()
        self.samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.latest_q: np.ndarray | None = None
        self.recording = False
        self.started = 0.0
        self.rejected = 0
        self.received = 0
        self.busy = False
        self.broad = broad
        self.create_subscription(JointState, "/joint_states", self.on_state, 100)
        self.create_subscription(Bool, "/dm_joint_target/busy", self.on_busy, 20)
        self.target_pub = self.create_publisher(Float64MultiArray, "/dm_joint_target", 10)

    def on_busy(self, msg: Bool) -> None:
        self.busy = bool(msg.data)

    def on_state(self, msg: JointState) -> None:
        self.received += 1
        if len(msg.position) != 6 or len(msg.effort) != 6:
            return
        try:
            order = [msg.name.index(name) for name in JOINTS]
        except ValueError:
            return
        q = np.asarray([msg.position[i] for i in order], dtype=float)
        tau = np.asarray([msg.effort[i] for i in order], dtype=float)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(tau)):
            self.rejected += 1
            return
        with self.lock:
            # Serial frames occasionally contain a motor value from another
            # feedback ID. Small mode has a tight envelope; broad mode uses
            # the controller limits and relies on the sequential-jump guard.
            outside_envelope = (
                np.any(q < LOWER - 0.05) or np.any(q > UPPER + 0.05)
                if self.broad
                else np.max(np.abs(q - CENTER)) > 0.60
            )
            if outside_envelope:
                self.rejected += 1
                return
            # Expected motion is below 0.5 rad/s. Reject isolated serial
            # feedback jumps without smearing them into the resampled log.
            if self.latest_q is not None and np.max(np.abs(q - self.latest_q)) > 0.08:
                self.rejected += 1
                return
            self.latest_q = q
            if self.recording:
                self.samples.append((time.monotonic() - self.started, q.copy(), tau.copy()))


def trajectory(duration: float, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration + 0.5 / rate, 1.0 / rate)
    window = np.sin(np.pi * t / duration) ** 2
    q = np.tile(CENTER, (len(t), 1))
    for joint in range(6):
        weights = rng.uniform(0.5, 1.0, 3)
        weights /= weights.sum()
        frequencies = rng.uniform(0.06, 0.24, 3)
        phases = rng.uniform(-np.pi, np.pi, 3)
        signal = np.zeros_like(t)
        initial = 0.0
        for weight, frequency, phase in zip(weights, frequencies, phases):
            term = weight * np.sin(2 * np.pi * frequency * t + phase)
            signal += term
            initial += weight * math.sin(phase)
        q[:, joint] += AMPLITUDE[joint] * window * (signal - initial)
    if np.any(q < LOWER) or np.any(q > UPPER):
        raise RuntimeError("generated trajectory exceeds driver joint limits")
    dq = np.gradient(q, t, axis=0)
    ddq = np.gradient(dq, t, axis=0)
    if np.max(np.abs(dq)) > 0.55 or np.max(np.abs(ddq)) > 2.5:
        raise RuntimeError("generated trajectory exceeds conservative speed/acceleration")
    return t, q


def wait_future(future, timeout: float):
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not future.done():
        raise TimeoutError("ROS action timed out")
    return future.result()


def wait_busy(node: Recorder, wanted: bool, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while node.busy != wanted and time.monotonic() < deadline:
        time.sleep(0.02)
    if node.busy != wanted:
        raise TimeoutError(f"/dm_joint_target/busy did not become {wanted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--trajectory-rate", type=float, default=25.0)
    parser.add_argument("--output-rate", type=float, default=100.0)
    parser.add_argument("--profile", choices=("small", "broad", "extended"), default="small")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    if args.duration < 20 or args.trajectory_rate < 10 or args.output_rate <= 0:
        parser.error("duration >=20 s, trajectory-rate >=10 Hz and output-rate >0 required")

    # The dedicated hardware-safe endpoint accepts one protected minimum-jerk
    # move at a time. One-second segments still resolve the <=0.24 Hz design.
    if args.profile == "small":
        _, q_cmd = trajectory(args.duration, 1.0, args.seed)
    elif args.profile == "broad":
        rng = np.random.default_rng(args.seed)
        targets = []
        for _ in range(args.cycles):
            for index in rng.permutation(len(SEARCH_POSES)):
                targets.extend((SEARCH_POSES[index], CENTER))
        q_cmd = np.asarray([CENTER, *targets])
    else:
        # Repeat the proven shoulder/elbow/wrist configurations with joint1
        # facing forward. Move through a known centre between configurations
        # and return to the original left observation centre at the end.
        rng = np.random.default_rng(args.seed)
        targets = []
        forward_poses = SEARCH_POSES.copy()
        forward_poses[:, 0] = 0.0
        for _ in range(args.cycles):
            for index in rng.permutation(len(SEARCH_POSES)):
                targets.extend((SEARCH_POSES[index], CENTER))
            targets.append(FORWARD_CENTER)
            for index in rng.permutation(len(forward_poses)):
                targets.extend((forward_poses[index], FORWARD_CENTER))
            targets.append(CENTER)
        q_cmd = np.asarray([CENTER, *targets])
    rclpy.init()
    node = Recorder(broad=args.profile != "small")
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while node.latest_q is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if node.latest_q is None:
            raise RuntimeError("no /joint_states with six effort values; rebuild/restart dm_controller")
        if np.max(np.abs(node.latest_q - CENTER)) > 0.60:
            raise RuntimeError(
                f"robot is not at observation pose; max error {np.max(np.abs(node.latest_q-CENTER)):.3f} rad"
            )
        deadline = time.monotonic() + 5.0
        while node.target_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        if node.target_pub.get_subscription_count() == 0:
            raise RuntimeError("/dm_joint_target safety endpoint unavailable")

        # Re-acquire the centre immediately before recording. The disabled MPC
        # path sends zero velocity, so geared joints can slowly relax after the
        # task manager declares arrival.
        wait_busy(node, False, 4.0)
        centre_msg = Float64MultiArray()
        centre_msg.data = [*CENTER.tolist(), 1.0]
        node.target_pub.publish(centre_msg)
        wait_busy(node, True, 2.0)
        wait_busy(node, False, 4.0)

        # Do not start physical excitation during a burst of corrupted serial
        # feedback. This catches the intermittent condition observed on the
        # shared STM32 link before the robot leaves the known centre pose.
        node.received = 0
        node.rejected = 0
        time.sleep(2.0)
        if node.received < 100 or node.rejected / node.received > 0.15:
            raise RuntimeError(
                "feedback preflight failed: "
                f"{node.rejected}/{node.received} joint-state frames rejected"
            )

        with node.lock:
            node.samples.clear()
            node.started = time.monotonic()
            node.recording = True
            node.received = 0
            node.rejected = 0
        print(
            f"STARTED: {args.profile} identification excitation "
            f"({len(q_cmd)-1} safe segments); do not touch the arm",
            flush=True,
        )
        for q in q_cmd[1:]:
            wait_busy(node, False, 4.0)
            msg = Float64MultiArray()
            segment_duration = 3.0 if args.profile != "small" else 1.0
            msg.data = [*q.tolist(), segment_duration]
            node.target_pub.publish(msg)
            wait_busy(node, True, 2.0)
            wait_busy(node, False, segment_duration + 3.0)
            if args.profile != "small":
                time.sleep(0.8)
        with node.lock:
            node.recording = False
            samples = list(node.samples)
    finally:
        node.recording = False
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)

    if len(samples) < args.duration * 20:
        raise RuntimeError(f"only {len(samples)} valid samples received")
    raw_t = np.asarray([row[0] for row in samples])
    raw_q = np.asarray([row[1] for row in samples])
    raw_tau = np.asarray([row[2] for row in samples])
    max_feedback_gap = float(np.max(np.diff(raw_t)))
    if np.any(np.abs(raw_tau) > 1.2 * EFFORT_LIMIT):
        raise RuntimeError("reported torque exceeds motor range")
    dt = 1.0 / args.output_rate
    t = np.arange(max(0.1, raw_t[0]), raw_t[-1] - 0.1, dt)
    q = np.column_stack([np.interp(t, raw_t, raw_q[:, i]) for i in range(6)])
    tau = np.column_stack([np.interp(t, raw_t, raw_tau[:, i]) for i in range(6)])

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", *[f"q{i}" for i in range(1, 7)], *[f"tau{i}" for i in range(1, 7)]])
        for k in range(len(t)):
            writer.writerow([f"{t[k]-t[0]:.9f}", *q[k], *tau[k]])
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.identification_log/v1",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "mapping_verified": True,
                "simulated": False,
                "sample_rate_hz": args.output_rate,
                "trajectory_seed": args.seed,
                "trajectory_duration_s": args.duration,
                "trajectory_profile": args.profile,
                "trajectory_segments": len(q_cmd) - 1,
                "segment_duration_s": 3.0 if args.profile != "small" else 1.0,
                "feedback_samples_rejected": node.rejected,
                "max_valid_feedback_gap_s": max_feedback_gap,
                "feedback_continuity_passed": max_feedback_gap <= 0.15,
                "position_units": "rad (URDF frame)",
                "torque_units": "Nm (joint-side, URDF sign)",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(t)} samples, rejected {node.rejected} feedback jumps)")


if __name__ == "__main__":
    main()
