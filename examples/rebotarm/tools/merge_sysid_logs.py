#!/usr/bin/env python3
"""Concatenate centre-to-centre SysID logs without mixing train/validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ["t", *[f"q{i}" for i in range(1, 7)], *[f"tau{i}" for i in range(1, 7)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+", help="independently recorded logs from the same split")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("at least two input logs are required")

    frames = []
    sources = []
    dt_reference = None
    previous_end = None
    time_offset = 0.0
    for name in args.inputs:
        path = Path(name).resolve()
        frame = pd.read_csv(path, float_precision="round_trip")
        missing = [column for column in COLUMNS if column not in frame]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frame = frame[COLUMNS].copy()
        values = frame.to_numpy(dtype=float)
        if len(frame) < 100 or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: insufficient or non-finite data")
        t = frame["t"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(t)))
        if np.any(np.diff(t) <= 0) or np.max(np.abs(np.diff(t) - dt)) > 1e-6:
            raise ValueError(f"{path}: timestamps are not uniform")
        if dt_reference is None:
            dt_reference = dt
        elif abs(dt - dt_reference) > 1e-9:
            raise ValueError("all logs must use the same sample period")
        start = frame[[f"q{i}" for i in range(1, 7)]].iloc[0].to_numpy(dtype=float)
        end = frame[[f"q{i}" for i in range(1, 7)]].iloc[-1].to_numpy(dtype=float)
        if previous_end is not None and np.max(np.abs(start - previous_end)) > 0.12:
            raise ValueError(f"{path}: boundary pose differs by more than 0.12 rad")
        frame["t"] = time_offset + np.arange(len(frame)) * dt_reference
        time_offset = float(frame["t"].iloc[-1] + dt_reference)
        previous_end = end
        frames.append(frame)
        sidecar = path.with_suffix(".json")
        sources.append(
            {
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "metadata": json.loads(sidecar.read_text()) if sidecar.exists() else None,
            }
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output, index=False, float_format="%.12g")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "rebotarm.merged_sysid_log/v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sources": sources,
                "sample_rate_hz": 1.0 / dt_reference,
                "rows": len(merged),
                "split_integrity_rule": "inputs belong to one split only; validation is recorded separately",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(merged)} rows from {len(frames)} logs)")


if __name__ == "__main__":
    main()
