"""reBotArm-specific FIGAROH data adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from figaroh.identification.base_identification import BaseIdentification
from figaroh.identification.identification_tools import (
    calculate_first_second_order_differentiation,
)


JOINTS = [f"joint{i}" for i in range(1, 7)]
Q_COLUMNS = [f"q{i}" for i in range(1, 7)]
TAU_COLUMNS = [f"tau{i}" for i in range(1, 7)]


class RebotArmIdentification(BaseIdentification):
    """Identify reBotArm dynamics from a canonical ``t,q1..q6,tau1..tau6`` CSV."""

    def __init__(
        self,
        robot: Any,
        config_file: str = "config/rebotarm_unified_config.yaml",
        data_file: str = "data/simulated/train.csv",
    ) -> None:
        super().__init__(robot, config_file)
        self.data_file = data_file
        self.active_joints = list(self.identif_config.get("active_joints") or JOINTS)
        if self.active_joints != JOINTS:
            raise ValueError(f"expected active joints {JOINTS}, got {self.active_joints}")

        joint_ids = [self.model.getJointId(name) for name in self.active_joints]
        if any(jid == 0 for jid in joint_ids):
            raise ValueError("one or more configured joints do not exist in the URDF")
        joints = [self.model.joints[jid] for jid in joint_ids]
        self.identif_config.update(
            active_joints=self.active_joints,
            act_Jid=joint_ids,
            act_J=joints,
            act_idxq=[joint.idx_q for joint in joints],
            act_idxv=[joint.idx_v for joint in joints],
            idx_act_joints=[jid - 1 for jid in joint_ids],
        )

    def load_trajectory_data(self, data_source: str = None) -> Dict[str, np.ndarray]:
        path = Path(data_source or self.data_file).expanduser().resolve()
        frame = pd.read_csv(path, float_precision="round_trip")
        required = ["t", *Q_COLUMNS, *TAU_COLUMNS]
        missing = [name for name in required if name not in frame]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        t = frame["t"].to_numpy(dtype=float)
        q = frame[Q_COLUMNS].to_numpy(dtype=float)
        tau = frame[TAU_COLUMNS].to_numpy(dtype=float)
        if len(frame) < 20:
            raise ValueError(f"{path}: need at least 20 samples")
        if not all(np.all(np.isfinite(x)) for x in (t, q, tau)):
            raise ValueError(f"{path}: t, q and tau must all be finite")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{path}: timestamps must be strictly increasing")

        dt = float(self.identif_config["ts"])
        jitter = np.diff(t) - dt
        if np.max(np.abs(jitter)) > max(1e-6, 0.01 * dt):
            raise ValueError(
                f"{path}: clock does not match configured {1 / dt:g} Hz "
                f"(max period error {np.max(np.abs(jitter)):.3g} s); resample first"
            )

        lower = np.asarray(self.model.lowerPositionLimit)
        upper = np.asarray(self.model.upperPositionLimit)
        bad = (q < lower - 1e-3) | (q > upper + 1e-3)
        if np.any(bad):
            row, joint = np.argwhere(bad)[0]
            raise ValueError(
                f"{path}: q{joint + 1}={q[row, joint]:.4f} at row {row} is outside "
                f"URDF limits [{lower[joint]:.4f}, {upper[joint]:.4f}]; check mapping"
            )

        q_kept, dq, ddq = calculate_first_second_order_differentiation(
            self.model, q, self.identif_config
        )
        count = len(q_kept)
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.trajectory_provenance = getattr(self, "trajectory_provenance", {})
        self.trajectory_provenance[str(path)] = {
            "file": str(path),
            "metadata_file": str(sidecar) if sidecar.exists() else None,
            "timing_source": "recorded",
            "position_units": "rad (URDF frame)",
            "effort_units": "Nm (joint-side, URDF sign)",
            "sample_period_s": dt,
            "source_rows": len(frame),
            "mapping_verified": metadata.get("mapping_verified"),
            "simulated": bool(metadata.get("simulated", False)),
        }
        return {
            "timestamps": t[:count].reshape(-1, 1),
            "positions": q_kept,
            "velocities": dq,
            "accelerations": ddq,
            "torques": tau[:count],
        }
