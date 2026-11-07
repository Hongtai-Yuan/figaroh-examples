#!/usr/bin/env python3
"""Use an identified URDF for mapped, limited reBotArm gravity feed-forward.

Without ``--execute`` this only computes an offline torque for ``--q``. Hardware
execution is opt-in, requires a verified mapping, ramps torque for two seconds,
and clamps each joint to 30% of its URDF effort limit by default.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


MOTOR_STATUS = {
    0: "disabled",
    1: "enabled",
    8: "over-voltage",
    9: "under-voltage",
    10: "over-current",
    11: "MOS over-temperature",
    12: "coil over-temperature",
    13: "communication lost",
    14: "overload",
}


def load_mapping(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    doc = yaml.safe_load(Path(path).read_text())
    if not doc.get("verified", False):
        raise ValueError(f"{path}: mapping is not verified")
    direction = np.asarray(doc["direction"], dtype=float)
    offset = np.asarray(doc["zero_offset_rad"], dtype=float)
    command_scale = np.asarray(doc.get("command_torque_scale", np.ones(6)), dtype=float)
    for name, value in (
        ("direction", direction),
        ("zero_offset_rad", offset),
        ("command_torque_scale", command_scale),
    ):
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"mapping {name} must contain six finite values")
    if not np.all(np.isin(direction, (-1, 1))):
        raise ValueError("direction must contain only -1 and +1")
    if np.any(command_scale <= 0):
        raise ValueError("command_torque_scale must be positive")
    return direction, offset, command_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, help="verified identified URDF")
    parser.add_argument("--mapping", default="config/hardware_mapping.yaml")
    parser.add_argument(
        "--gravity-calibration",
        default=None,
        help="accepted static_gravity_model/v1 JSON with conservative joint scales",
    )
    parser.add_argument("--q", nargs=6, type=float, default=[0, 1.55, 1.72, 0, 0, 0])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-unvalidated-hardware-test", action="store_true",
        help="explicitly permit staged hardware validation of an offline-only calibration",
    )
    parser.add_argument("--arm-config", default=None)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="live-test duration in seconds; zero runs until interrupted")
    parser.add_argument("--compensation-gain", type=float, default=1.0,
                        help="fraction of identified gravity torque to apply")
    parser.add_argument("--torque-limit-fraction", type=float, default=0.30)
    parser.add_argument("--hold-kp", type=float, default=0.0,
                        help="small MIT position stiffness around the starting pose")
    parser.add_argument(
        "--damping-kd", type=float, nargs="+", default=[1.0],
        help="one MIT damping value for all joints, or six joint-specific values",
    )
    parser.add_argument(
        "--max-position-deviation", type=float, default=0.35,
        help="maximum displacement from start; zero disables this in favor of absolute limits",
    )
    parser.add_argument(
        "--max-velocity", type=float, nargs="+", default=[1.0],
        help="one velocity safety limit for all joints, or six joint-specific limits",
    )
    parser.add_argument("--rate", type=float, default=200.0)
    args = parser.parse_args()
    if not 0 < args.torque_limit_fraction <= 1 or args.rate <= 0:
        parser.error("torque limit fraction must be in (0,1] and rate must be positive")
    if not 0 <= args.compensation_gain <= 1:
        parser.error("compensation gain must be in [0,1]")
    damping_values = np.asarray(args.damping_kd, dtype=float)
    if damping_values.size == 1:
        damping_values = np.full(6, float(damping_values[0]))
    if damping_values.shape != (6,) or np.any(damping_values <= 0) or np.any(damping_values > 5):
        parser.error("damping-kd must contain one or six values, each in (0,5]")
    velocity_limits = np.asarray(args.max_velocity, dtype=float)
    if velocity_limits.size == 1:
        velocity_limits = np.full(6, float(velocity_limits[0]))
    if velocity_limits.shape != (6,) or np.any(velocity_limits <= 0):
        parser.error("max-velocity must contain one or six positive values")
    if args.duration < 0 or args.hold_kp < 0:
        parser.error("duration and hold-kp must be non-negative")
    if args.max_position_deviation < 0:
        parser.error("position deviation must be non-negative")

    direction, offset, command_scale = load_mapping(args.mapping)
    model = pin.buildModelFromUrdf(str(Path(args.urdf).resolve()))
    data = model.createData()
    urdf_velocity_limits = np.asarray(model.velocityLimit, dtype=float)
    if np.any(velocity_limits > urdf_velocity_limits + 1e-9):
        parser.error(
            "max-velocity exceeds URDF limits: requested "
            f"{velocity_limits.tolist()}, URDF {urdf_velocity_limits.tolist()}"
        )
    effort = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
    gravity_scale = np.ones(6)
    calibrated_q_min = None
    calibrated_q_max = None
    if args.gravity_calibration:
        calibration = json.loads(Path(args.gravity_calibration).read_text())
        if calibration.get("schema") != "rebotarm.static_gravity_model/v1":
            raise ValueError("unsupported gravity calibration schema")
        if not calibration.get("accepted", False):
            raise ValueError("gravity calibration was not accepted")
        if (
            args.execute
            and calibration.get("deployment_status") != "static_hold_test_passed"
            and not args.allow_unvalidated_hardware_test
        ):
            raise ValueError(
                "calibration has not passed a static hardware test; use the explicit "
                "hardware-test override only for staged validation"
            )
        expected_hash = calibration.get("urdf_sha256")
        actual_hash = __import__("hashlib").sha256(Path(args.urdf).read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("gravity calibration was fitted for a different URDF")
        gravity_scale = np.asarray(calibration["gravity_scale"], dtype=float)
        if gravity_scale.shape != (6,) or not np.all(np.isfinite(gravity_scale)):
            raise ValueError("gravity_scale must contain six finite values")
        calibrated_q_min = np.asarray(calibration["calibrated_q_min_rad"], dtype=float)
        calibrated_q_max = np.asarray(calibration["calibrated_q_max_rad"], dtype=float)

    # RobotArm exposes raw motor coordinates. Its zero/direction convention is
    # not the URDF's one-sided J2/J3 convention, so enforce the same symmetric
    # command envelope used by the current ROS driver in motor coordinates.
    safe_motor_lower = np.array([-3.00, -3.00, -3.40, -2.50, -2.10, -3.00])
    safe_motor_upper = np.array([ 3.00,  1.50,  1.50,  2.10,  2.10,  3.00])

    def gravity_motor(q_motor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_urdf = direction * q_motor + offset
        tau_urdf = gravity_scale * pin.computeGeneralizedGravity(model, data, q_urdf).copy()
        tau_motor = args.compensation_gain * direction * command_scale * tau_urdf
        limit = args.torque_limit_fraction * effort
        return q_urdf, np.clip(tau_motor, -limit, limit)

    if not args.execute:
        q_urdf = np.asarray(args.q, dtype=float)
        q_motor = direction * (q_urdf - offset)
        _, tau = gravity_motor(q_motor)
        print("offline only; no hardware was opened")
        print("q_urdf:", np.array2string(q_urdf, precision=4))
        print("tau_motor_command:", np.array2string(tau, precision=4), "Nm")
        return

    workspace = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(workspace / "vendor" / "reBotArm_control_py"))
    from reBotArm_control_py.actuator import RobotArm

    def wait_for_stable_feedback(arm, timeout: float = 4.0) -> np.ndarray:
        """Require all six real motor states and a stable rolling position window."""
        samples: list[np.ndarray] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            arm._request_and_poll()
            states = [arm._motor_map[name].get_state() for name in arm.joint_names]
            if all(state is not None for state in states):
                q = np.asarray([state.pos for state in states], dtype=float)
                # The calibrated/validated workspace never approaches the raw
                # Damiao +/-12.5 rad encoding limit. Reject mode acknowledgements
                # or torn frames that may temporarily decode near that limit.
                if np.all(np.isfinite(q)) and np.all(np.abs(q) <= 3.2):
                    samples.append(q)
                    samples = samples[-8:]
                    if len(samples) == 8:
                        window = np.asarray(samples)
                        if float(np.max(np.ptp(window, axis=0))) <= 0.02:
                            return np.median(window, axis=0)
                else:
                    samples.clear()
            time.sleep(0.02)
        raise RuntimeError("six-joint feedback did not become stable")

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    arm = RobotArm(args.arm_config)
    period = 1.0 / args.rate
    try:
        q_pre_enable = wait_for_stable_feedback(arm)
        arm.enable()
        hold_kp = np.full(6, args.hold_kp)
        damping_kd = damping_values.copy()
        if not arm.mode_mit(kp=hold_kp, kd=damping_kd):
            raise RuntimeError("not all motors entered MIT mode")
        arm.mit(
            pos=q_pre_enable, vel=np.zeros(6), kp=hold_kp, kd=damping_kd,
            tau=np.zeros(6), request_feedback=True,
        )
        q_reference = wait_for_stable_feedback(arm)
        started = time.monotonic()
        deadline = started
        last_report = started
        print("LIVE TEST STARTED")
        print("q_start:", np.array2string(q_reference, precision=4))
        velocity_violation_count = 0
        while running:
            q_motor, velocity, _ = arm.get_state()
            states = [arm._motor_map[name].get_state() for name in arm.joint_names]
            statuses = [None if state is None else int(state.status_code) for state in states]
            bad_status = [i for i, status in enumerate(statuses) if status != 1]
            if bad_status:
                detail = ", ".join(
                    f"J{i+1}={statuses[i]}({MOTOR_STATUS.get(statuses[i], 'unknown')})"
                    for i in bad_status
                )
                raise RuntimeError(f"motor status fault: {detail}; all={statuses}")
            if not np.all(np.isfinite(q_motor)) or not np.all(np.isfinite(velocity)):
                raise RuntimeError("non-finite motor feedback")
            deviation_vector = np.abs(q_motor - q_reference)
            speed_vector = np.abs(velocity)
            deviation_joint = int(np.argmax(deviation_vector))
            speed_joint = int(np.argmax(speed_vector))
            deviation = float(deviation_vector[deviation_joint])
            speed = float(speed_vector[speed_joint])
            outside = np.flatnonzero(
                (q_motor < safe_motor_lower) | (q_motor > safe_motor_upper)
            )
            if outside.size:
                joint = int(outside[0])
                raise RuntimeError(
                    f"absolute motor-coordinate limit exceeded: J{joint+1}={q_motor[joint]:.3f} "
                    f"outside [{safe_motor_lower[joint]:.3f}, {safe_motor_upper[joint]:.3f}] rad"
                )
            if args.max_position_deviation and deviation > args.max_position_deviation:
                raise RuntimeError(
                    f"position safety limit exceeded: J{deviation_joint+1} "
                    f"{deviation:.3f} rad; q={np.array2string(q_motor, precision=3)}"
                )
            velocity_excess = speed_vector > velocity_limits
            if np.any(velocity_excess):
                velocity_violation_count += 1
                speed_joint = int(np.argmax(speed_vector / velocity_limits))
                if velocity_violation_count >= 3:
                    raise RuntimeError(
                        f"velocity safety limit exceeded for 3 frames: J{speed_joint+1} "
                        f"{speed_vector[speed_joint]:.3f} > {velocity_limits[speed_joint]:.3f} rad/s; "
                        f"dq={np.array2string(velocity, precision=3)}"
                    )
                # Do not calculate a new torque from an over-speed sample. Keep
                # the previous command for at most two 100 Hz frames.
                time.sleep(period)
                continue
            velocity_violation_count = 0
            _, tau = gravity_motor(q_motor)
            ramp = min(1.0, (time.monotonic() - started) / 2.0)
            arm.mit(
                pos=q_reference,
                vel=np.zeros(6),
                kp=hold_kp,
                kd=damping_kd,
                tau=ramp * tau,
                request_feedback=True,
            )
            now = time.monotonic()
            if now - last_report >= 1.0:
                print(
                    f"t={now-started:.1f}s max_dq={deviation:.4f} "
                    f"max_dqdt={speed:.4f} status={statuses} "
                    f"tau={np.array2string(ramp*tau, precision=3)}",
                    flush=True,
                )
                last_report = now
            if args.duration and now - started >= args.duration:
                break
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        arm.disconnect()
        print("LIVE TEST STOPPED; MOTORS DISABLED")


if __name__ == "__main__":
    main()
