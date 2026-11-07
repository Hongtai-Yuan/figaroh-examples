#!/usr/bin/env python3
"""Map raw motor feedback into a uniform, URDF-frame FIGAROH CSV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def vector(mapping: dict, key: str) -> np.ndarray:
    value = np.asarray(mapping.get(key), dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError(f"mapping {key!r} must contain six finite values")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--mapping", default="config/hardware_mapping.yaml")
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument(
        "--allow-unverified-mapping",
        action="store_true",
        help="for mapping diagnostics only; resulting metadata remains unverified",
    )
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("rate must be positive")

    mapping_path = Path(args.mapping)
    mapping = yaml.safe_load(mapping_path.read_text())
    verified = bool(mapping.get("verified", False))
    if not verified and not args.allow_unverified_mapping:
        raise SystemExit(
            f"{mapping_path} is not verified; check every joint direction/zero/torque "
            "scale, set verified: true, or use --allow-unverified-mapping only for diagnostics"
        )
    direction = vector(mapping, "direction")
    if not np.all(np.isin(direction, (-1, 1))):
        raise ValueError("mapping direction entries must be exactly -1 or +1")
    offset = vector(mapping, "zero_offset_rad")
    scale = vector(mapping, "torque_scale")

    raw = pd.read_csv(args.raw_csv)
    required = ["t_raw"]
    required += [f"q_motor{i}" for i in range(1, 7)]
    required += [f"tau_motor{i}" for i in range(1, 7)]
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"raw log is missing columns {missing}")
    t_raw = raw["t_raw"].to_numpy(dtype=float)
    if len(t_raw) < 20 or np.any(np.diff(t_raw) <= 0):
        raise ValueError("raw timestamps need at least 20 strictly increasing samples")
    q_raw = raw[[f"q_motor{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    tau_raw = raw[[f"tau_motor{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    q_mapped = q_raw * direction + offset
    tau_mapped = tau_raw * direction * scale

    dt = 1.0 / args.rate
    t = np.arange(t_raw[0], t_raw[-1] - 0.5 * dt, dt)
    q = np.column_stack([np.interp(t, t_raw, q_mapped[:, i]) for i in range(6)])
    tau = np.column_stack([np.interp(t, t_raw, tau_mapped[:, i]) for i in range(6)])
    out = pd.DataFrame({"t": t - t[0]})
    for i in range(6):
        out[f"q{i + 1}"] = q[:, i]
    for i in range(6):
        out[f"tau{i + 1}"] = tau[:, i]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, float_format="%.12g")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.identification_log/v1",
                "converted_at": datetime.now(timezone.utc).isoformat(),
                "source": str(Path(args.raw_csv).resolve()),
                "mapping": str(mapping_path.resolve()),
                "mapping_verified": verified,
                "sample_rate_hz": args.rate,
                "position_units": "rad (URDF frame)",
                "torque_units": "Nm (joint-side, URDF sign)",
                "resampling": "linear",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(out)} uniform samples at {args.rate:g} Hz)")


if __name__ == "__main__":
    main()
