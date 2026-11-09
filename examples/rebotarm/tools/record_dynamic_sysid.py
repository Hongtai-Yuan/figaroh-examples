#!/usr/bin/env python3
"""Execute a prevalidated SysID trajectory and record synchronized q/tau."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint


JOINTS = [f"joint{i}" for i in range(1, 7)]
CENTER = np.array([0.0, 0.5235987756, 0.7853981634, -0.2508652382, 0.0, 0.0])
Q_LOWER = np.array([-0.60, 0.18, 0.32, -0.70, -0.90, -0.90])
Q_UPPER = np.array([0.60, 0.92, 1.25, 0.20, 0.90, 0.90])
MONITOR_DQ = np.array([0.85, 0.85, 0.85, 1.10, 1.25, 1.40])
EFFORT_LIMIT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])


def wait_until(predicate, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not predicate():
        raise TimeoutError(message)


class DynamicRecorder(Node):
    def __init__(self) -> None:
        super().__init__("rebotarm_dynamic_sysid_recorder")
        self.lock = threading.Lock()
        self.latest_q = None
        self.last_valid_q = None
        self.last_valid_wall = 0.0
        self.samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self.recording = False
        self.recording_started = 0.0
        self.received = 0
        self.rejected = 0
        self.violation_reason: str | None = None
        self.speed_violation_count = 0
        self.effort_violation_count = 0
        self.busy = False
        self.create_subscription(JointState, "/joint_states", self.on_state, 200)
        self.create_subscription(Bool, "/dm_joint_target/busy", self.on_busy, 20)
        self.target_pub = self.create_publisher(Float64MultiArray, "/dm_joint_target", 10)
        self.action = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")

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
        if not all(np.all(np.isfinite(value)) for value in (q, dq, tau)):
            self.rejected += 1
            return
        now = time.monotonic()
        with self.lock:
            if self.last_valid_q is not None and np.max(np.abs(q - self.last_valid_q)) > 0.10:
                self.rejected += 1
                return
            self.last_valid_q = q
            self.latest_q = q
            self.last_valid_wall = now
            outside = np.flatnonzero((q < Q_LOWER - 0.03) | (q > Q_UPPER + 0.03))
            if outside.size:
                joint = int(outside[0])
                self.violation_reason = f"J{joint+1} measured position {q[joint]:.3f} outside monitor box"
            self.speed_violation_count = (
                self.speed_violation_count + 1
                if np.any(np.abs(dq) > MONITOR_DQ) else 0
            )
            self.effort_violation_count = (
                self.effort_violation_count + 1
                if np.any(np.abs(tau) > 1.10 * EFFORT_LIMIT) else 0
            )
            if self.speed_violation_count >= 3:
                joint = int(np.argmax(np.abs(dq) / MONITOR_DQ))
                self.violation_reason = f"J{joint+1} measured speed {dq[joint]:.3f} exceeds monitor limit"
            if self.effort_violation_count >= 3:
                joint = int(np.argmax(np.abs(tau) / EFFORT_LIMIT))
                self.violation_reason = f"J{joint+1} measured effort {tau[joint]:.3f} exceeds motor range"
            if self.recording:
                self.samples.append((now - self.recording_started, q.copy(), dq.copy(), tau.copy()))


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(f"missing trajectory metadata {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "rebotarm.safe_sysid_trajectory/v1":
        raise ValueError("unsupported or unvalidated trajectory schema")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get("trajectory_sha256"):
        raise ValueError("trajectory CSV hash does not match metadata")
    frame = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(frame["t"], dtype=float)
    q = np.column_stack([frame[f"q{i}"] for i in range(1, 7)]).astype(float)
    if len(t) < 100 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("trajectory needs finite, strictly increasing timestamps")
    if q.shape != (len(t), 6) or np.any(~np.isfinite(q)):
        raise ValueError("trajectory needs six finite joint positions")
    if t[0] != 0.0 or t[-1] > 120.0:
        raise ValueError("trajectory must start at t=0 and last <=120 seconds")
    centre = np.asarray(metadata.get("centre_rad"), dtype=float)
    if centre.shape != (6,) or np.any(~np.isfinite(centre)):
        raise ValueError("trajectory metadata needs a finite six-axis centre")
    if np.any(centre < Q_LOWER) or np.any(centre > Q_UPPER):
        raise ValueError("trajectory centre exceeds the recorder safety box")
    if np.max(np.abs(q[0] - centre)) > 1e-6 or np.max(np.abs(q[-1] - centre)) > 1e-6:
        raise ValueError("trajectory must start and end at the safe centre")
    if np.any(q < Q_LOWER) or np.any(q > Q_UPPER):
        raise ValueError("trajectory position exceeds the recorder safety box")
    return t, q, centre, metadata


def move_to_centre(node: DynamicRecorder, centre: np.ndarray) -> None:
    wait_until(lambda: not node.busy, 4.0, "joint target remained busy")
    msg = Float64MultiArray()
    msg.data = [*centre.tolist(), 4.0, 0.01]
    node.target_pub.publish(msg)
    wait_until(lambda: node.busy, 2.0, "centre move was not accepted")
    wait_until(lambda: not node.busy, 9.0, "centre move did not finish")
    wait_until(
        lambda: node.latest_q is not None and np.max(np.abs(node.latest_q - centre)) <= 0.08,
        2.0,
        "robot did not reach the safe centre",
    )


def make_goal(t: np.ndarray, q: np.ndarray) -> FollowJointTrajectory.Goal:
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINTS
    for stamp, position in zip(t, q):
        point = JointTrajectoryPoint()
        point.positions = position.tolist()
        seconds = int(stamp)
        nanoseconds = int(round((float(stamp) - seconds) * 1e9))
        if nanoseconds == 1_000_000_000:
            seconds += 1
            nanoseconds = 0
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds
        goal.trajectory.points.append(point)
    return goal


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-rate", type=float, default=100.0)
    args = parser.parse_args()
    if not 20.0 <= args.output_rate <= 300.0:
        parser.error("output-rate must be in [20,300] Hz")
    return args


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectory).resolve()
    output = Path(args.output).resolve()
    t_cmd, q_cmd, centre, trajectory_metadata = load_trajectory(trajectory_path)
    rclpy.init()
    node = DynamicRecorder()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    goal_handle = None
    success = False
    try:
        wait_until(lambda: node.latest_q is not None, 5.0, "no complete /joint_states feedback")
        wait_until(lambda: node.target_pub.get_subscription_count() > 0, 5.0, "/dm_joint_target unavailable")
        if not node.action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("FollowJointTrajectory action server unavailable")
        move_to_centre(node, centre)

        # Reject a noisy/stale link before enabling dynamic motion.
        with node.lock:
            node.received = node.rejected = 0
            node.violation_reason = None
        time.sleep(2.0)
        with node.lock:
            received, rejected = node.received, node.rejected
            feedback_age = time.monotonic() - node.last_valid_wall
        if received < 200 or rejected / max(received, 1) > 0.03 or feedback_age > 0.10:
            raise RuntimeError(
                f"feedback preflight failed: received={received}, rejected={rejected}, age={feedback_age:.3f}s"
            )

        with node.lock:
            node.samples.clear()
            node.recording_started = time.monotonic()
            node.recording = True
            node.received = node.rejected = 0
            node.violation_reason = None
        future = node.action.send_goal_async(make_goal(t_cmd, q_cmd))
        wait_until(future.done, 5.0, "trajectory goal response timed out")
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("trajectory was rejected by the hardened driver")
        print(f"STARTED: {t_cmd[-1]:.1f}s dynamic SysID trajectory; do not touch the arm", flush=True)
        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + float(t_cmd[-1]) + 8.0
        while not result_future.done():
            with node.lock:
                reason = node.violation_reason
                feedback_age = time.monotonic() - node.last_valid_wall
            if reason or feedback_age > 0.12 or time.monotonic() > deadline:
                goal_handle.cancel_goal_async()
                if reason:
                    raise RuntimeError(reason)
                if feedback_age > 0.12:
                    raise RuntimeError(f"feedback became stale for {feedback_age:.3f}s")
                raise TimeoutError("trajectory result timed out")
            time.sleep(0.01)
        result = result_future.result().result
        if result.error_code != 0:
            raise RuntimeError(f"trajectory action failed: {result.error_string}")
        success = True
        with node.lock:
            node.recording = False
            samples = list(node.samples)
            received, rejected = node.received, node.rejected
        print("COMPLETED: dynamic trajectory returned to safe centre", flush=True)
    finally:
        with node.lock:
            node.recording = False
        if goal_handle is not None and not success:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()

    if len(samples) < t_cmd[-1] * 100:
        raise RuntimeError(f"only {len(samples)} valid feedback samples recorded")
    raw_t = np.asarray([sample[0] for sample in samples])
    raw_q = np.asarray([sample[1] for sample in samples])
    raw_tau = np.asarray([sample[3] for sample in samples])
    max_gap = float(np.max(np.diff(raw_t)))
    if max_gap > 0.08:
        raise RuntimeError(f"feedback gap {max_gap:.3f}s is too large")
    dt = 1.0 / args.output_rate
    t = np.arange(raw_t[0], raw_t[-1] - 0.5 * dt, dt)
    q = np.column_stack([np.interp(t, raw_t, raw_q[:, i]) for i in range(6)])
    tau = np.column_stack([np.interp(t, raw_t, raw_tau[:, i]) for i in range(6)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", *[f"q{i}" for i in range(1, 7)], *[f"tau{i}" for i in range(1, 7)]])
        for k in range(len(t)):
            writer.writerow([f"{t[k]-t[0]:.9f}", *q[k], *tau[k]])
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.dynamic_sysid_log/v1",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "trajectory": str(trajectory_path),
                "trajectory_sha256": trajectory_metadata["trajectory_sha256"],
                "trajectory_seed": trajectory_metadata["seed"],
                "sample_rate_hz": args.output_rate,
                "feedback_frames_received": received,
                "feedback_frames_rejected": rejected,
                "max_valid_feedback_gap_s": max_gap,
                "position_units": "rad (URDF frame)",
                "torque_units": "motor-reported Nm (URDF sign)",
                "completed_and_returned_to_centre": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(t)} samples, rejected {rejected} frames)")


if __name__ == "__main__":
    main()
