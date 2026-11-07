#!/usr/bin/env python3
"""Record paired, stationary gravity-effort measurements on the real arm."""

from __future__ import annotations

import argparse
import csv
import json
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
FORWARD_CENTER = np.array([0.0, 0.5235987756, 0.7853981634, -0.2508652382, 0.0, 0.0])
KNOWN = np.array(
    [
        FORWARD_CENTER,
        [0.0, 0.549779, 0.706858, -0.250865, -1.195551, 0.0],
        [0.0, 0.130900, 0.280998, 0.141834, 0.043633, 0.043633],
        [0.0, 0.916298, 1.289799, -0.643564, -0.043633, -0.043633],
        [0.0, 0.497419, 0.863938, -0.250865, 1.195551, 0.0],
    ],
    dtype=float,
)
PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4))
TRAIN_POSES = np.vstack((KNOWN, np.array([(KNOWN[a] + KNOWN[b]) / 2 for a, b in PAIRS])))
VALIDATION_POSES = np.vstack(
    [
        (2 * KNOWN[a] + KNOWN[b]) / 3
        for a, b in PAIRS[:5]
    ]
    + [
        (KNOWN[a] + 2 * KNOWN[b]) / 3
        for a, b in PAIRS[:5]
    ]
)


def single_joint_sweep() -> np.ndarray:
    poses = [FORWARD_CENTER.copy()]
    for joint, values in (
        (1, (0.25, 0.40, 0.68, 0.82)),
        (2, (0.45, 0.62, 0.95, 1.12)),
        (3, (-0.55, -0.40, -0.10, 0.10)),
        (4, (-1.20, -0.80, -0.40, 0.40, 0.80, 1.20)),
    ):
        for value in values:
            pose = FORWARD_CENTER.copy()
            pose[joint] = value
            poses.append(pose)
    return np.asarray(poses)


TARGETED_POSES = single_joint_sweep()
APPROACH_DELTA = np.array([0.0, 0.055, 0.055, 0.040, 0.050, 0.0])


class StaticRecorder(Node):
    def __init__(self) -> None:
        super().__init__("rebotarm_static_gravity_recorder")
        self.lock = threading.Lock()
        self.latest_q = None
        self.latest_dq = None
        self.latest_tau = None
        self.last_valid_q = None
        self.window = []
        self.collecting = False
        self.busy = False
        self.received = 0
        self.rejected = 0
        self.create_subscription(JointState, "/joint_states", self.on_state, 100)
        self.create_subscription(Bool, "/dm_joint_target/busy", self.on_busy, 20)
        self.target_pub = self.create_publisher(Float64MultiArray, "/dm_joint_target", 10)

    def on_busy(self, msg: Bool) -> None:
        self.busy = bool(msg.data)

    def on_state(self, msg: JointState) -> None:
        self.received += 1
        if len(msg.position) != 6 or len(msg.velocity) != 6 or len(msg.effort) != 6:
            self.rejected += 1
            return
        try:
            order = [msg.name.index(name) for name in JOINTS]
        except ValueError:
            self.rejected += 1
            return
        q = np.asarray([msg.position[i] for i in order], dtype=float)
        dq = np.asarray([msg.velocity[i] for i in order], dtype=float)
        tau = np.asarray([msg.effort[i] for i in order], dtype=float)
        if not all(np.all(np.isfinite(x)) for x in (q, dq, tau)):
            self.rejected += 1
            return
        with self.lock:
            if self.last_valid_q is not None and np.max(np.abs(q - self.last_valid_q)) > 0.08:
                self.rejected += 1
                return
            self.last_valid_q = q
            self.latest_q, self.latest_dq, self.latest_tau = q, dq, tau
            if self.collecting:
                self.window.append((q.copy(), dq.copy(), tau.copy()))


def wait_busy(node: StaticRecorder, wanted: bool, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while node.busy != wanted and time.monotonic() < deadline:
        time.sleep(0.02)
    if node.busy != wanted:
        raise TimeoutError(f"/dm_joint_target/busy did not become {wanted}")


def move(node: StaticRecorder, target: np.ndarray, duration: float) -> None:
    wait_busy(node, False, 4.0)
    msg = Float64MultiArray()
    # The optional eighth field requests the driver's calibration-only precise
    # completion tolerance. Ordinary /dm_joint_target callers remain at 0.06.
    msg.data = [*target.tolist(), duration, 0.003]
    node.target_pub.publish(msg)
    wait_busy(node, True, 2.0)
    wait_busy(node, False, duration + 4.0)


def measure(node: StaticRecorder, target: np.ndarray, seconds: float):
    with node.lock:
        node.window.clear()
        node.collecting = True
    time.sleep(seconds)
    with node.lock:
        node.collecting = False
        window = list(node.window)
    if len(window) < max(30, int(seconds * 50)):
        raise RuntimeError(f"only {len(window)} valid stationary samples")
    q = np.asarray([x[0] for x in window])
    dq = np.asarray([x[1] for x in window])
    tau = np.asarray([x[2] for x in window])
    stationary = (np.max(np.abs(dq), axis=1) < 0.08) & (
        np.max(np.abs(q - target), axis=1) < 0.08
    )
    if np.count_nonzero(stationary) < 30:
        raise RuntimeError("joint did not settle at static target")
    q, tau = q[stationary], tau[stationary]
    return np.median(q, axis=0), np.median(tau, axis=0), np.std(q, axis=0), np.std(tau, axis=0), len(q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument(
        "--pose-set", choices=("train", "validation", "targeted"), default="train"
    )
    parser.add_argument("--settle", type=float, default=1.2)
    parser.add_argument("--measure", type=float, default=1.2)
    parser.add_argument("--max-poses", type=int, default=None)
    args = parser.parse_args()
    if args.settle < 0.8 or args.measure < 0.8:
        parser.error("settle and measure must each be at least 0.8 seconds")
    poses = {
        "train": TRAIN_POSES,
        "validation": VALIDATION_POSES,
        "targeted": TARGETED_POSES,
    }[args.pose_set]

    rclpy.init()
    node = StaticRecorder()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    rows = []
    failure = None
    try:
        deadline = time.monotonic() + 5.0
        while node.latest_q is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if node.latest_q is None:
            raise RuntimeError("no six-axis position/velocity/effort feedback")
        deadline = time.monotonic() + 5.0
        while node.target_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        if node.target_pub.get_subscription_count() == 0:
            raise RuntimeError("/dm_joint_target safety endpoint unavailable")

        move(node, FORWARD_CENTER, 4.0)
        node.received = node.rejected = 0
        time.sleep(2.0)
        if node.received < 100 or node.rejected / node.received > 0.05:
            raise RuntimeError(
                f"feedback preflight failed: {node.rejected}/{node.received} frames rejected"
            )
        node.received = node.rejected = 0
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(poses))
        if args.max_poses is not None:
            if args.max_poses < 1:
                raise ValueError("--max-poses must be positive")
            order = order[: args.max_poses]
        print(f"STARTED: {len(order)} poses x 2 approach directions", flush=True)
        for sequence, pose_id in enumerate(order, 1):
            target = poses[pose_id]
            for approach in (-1, 1):
                pre = target + approach * APPROACH_DELTA
                move(node, pre, 1.5)
                move(node, target, 1.0)
                time.sleep(args.settle)
                q, tau, q_std, tau_std, count = measure(node, target, args.measure)
                rows.append((int(pose_id), approach, count, q, tau, q_std, tau_std))
            print(f"pose {sequence}/{len(order)} complete", flush=True)
    except Exception as exc:
        failure = exc
    finally:
        node.collecting = False
        try:
            if node.target_pub.get_subscription_count() > 0:
                move(node, FORWARD_CENTER, 4.0)
        except Exception:
            pass
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    if failure is not None:
        raise failure

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pose_id", "approach", "samples"]
    columns += [f"q{i}" for i in range(1, 7)] + [f"tau{i}" for i in range(1, 7)]
    columns += [f"q_std{i}" for i in range(1, 7)] + [f"tau_std{i}" for i in range(1, 7)]
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for pose_id, approach, count, q, tau, q_std, tau_std in rows:
            writer.writerow([pose_id, approach, count, *q, *tau, *q_std, *tau_std])
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.static_gravity/v1",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "payload": "no added calibration mass",
                "pose_set": args.pose_set,
                "pose_count": len(order),
                "paired_approaches": True,
                "feedback_frames_received": node.received,
                "feedback_frames_rejected": node.rejected,
                "mapping_verified": True,
                "position_units": "rad (URDF frame)",
                "torque_units": "motor-reported Nm (URDF sign)",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(rows)} paired-direction rows; rejected {node.rejected} frames)")


if __name__ == "__main__":
    main()
