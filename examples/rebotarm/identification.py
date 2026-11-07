#!/usr/bin/env python3
"""Run reBotArm dynamic identification with independent validation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.rebotarm.utils.rebotarm_tools import RebotArmIdentification  # noqa: E402
from examples.verification import add_verification_args, run_verification  # noqa: E402
from figaroh.tools.robot import load_robot  # noqa: E402
from figaroh.tools.run_archive import archive_run, compute_run_dir  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rebotarm_simulated.yaml")
    parser.add_argument("--urdf", default="urdf/rebotarm.urdf")
    parser.add_argument("--data", default="data/simulated/train.csv")
    parser.add_argument("--validation", default=None, help="independently recorded CSV")
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--operator", default=None)
    parser.add_argument("--wls", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--html-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--archive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--export-urdf",
        default=None,
        metavar="PATH",
        help="export only an accepted reconstructed physical model",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    add_verification_args(parser)
    return parser.parse_args(argv)


def run_identification(args: argparse.Namespace) -> RebotArmIdentification:
    for value, label in ((args.urdf, "URDF"), (args.config, "config"), (args.data, "data")):
        if not Path(value).exists():
            raise FileNotFoundError(f"{label} not found: {value}")
    robot = load_robot(args.urdf, package_dirs="../../models", load_by_urdf=True)
    ident = RebotArmIdentification(robot, args.config, args.data)
    if args.validation:
        if Path(args.validation).resolve() == Path(args.data).resolve():
            raise ValueError("training and validation must be independent files")
        ident.identif_config["validation_data_file"] = args.validation
    instance = dict(ident.identif_config.get("instance") or {})
    if args.asset_id:
        instance["asset_id"] = args.asset_id
    if args.operator:
        instance["operator"] = args.operator
    ident.identif_config["instance"] = instance
    wls = args.wls if args.wls is not None else ident.identif_config.get("wls", False)
    ident.initialize()
    ident.solve(
        decimate=False,
        plotting=args.plot,
        save_results=False,
        html_report=False,
        wls=wls,
    )
    return ident


def main(args: argparse.Namespace) -> None:
    ident = run_identification(args)
    print("\nreBotArm dynamic identification")
    print(f"  training:        {Path(args.data).resolve()}")
    print(f"  validation:      {ident.identif_config.get('validation_data_file')}")
    print(f"  base parameters: {len(ident.params_base)}")
    print(f"  condition no.:   {ident.result['condition number']:.3g}")
    print(f"  correlation:     {ident.correlation:.6f}")
    for name, value in zip(ident.params_base, ident.phi_base):
        print(f"    {name:40s} {value:+.8g}")

    run_dir = compute_run_dir(ident) if (args.html_report or args.verify or args.archive) else None
    if args.html_report:
        ident.export_html_report(output_path=str(run_dir / "report.html"))
    verdict = None
    if args.verify:
        verdict = run_verification(
            ident, run_dir, args.verification_scope, args.acceptance_profile
        )
    if args.export_urdf and (
        verdict is None
        or not verdict.passed
        or args.verification_scope != "prediction"
        or args.acceptance_profile is None
    ):
        raise RuntimeError(
            "refusing URDF export without passing prediction verification on "
            "independent data and an explicit --acceptance-profile"
        )
    if args.export_urdf:
        ident.export_urdf(args.urdf, output_path=args.export_urdf)
        print(f"  exported URDF:   {Path(args.export_urdf).resolve()}")
    if args.archive:
        archive_run(ident, run_dir)
    if run_dir:
        print(f"  run directory:   {run_dir}")
    if verdict is not None and not verdict.passed:
        raise SystemExit(f"verification {verdict.status} ({verdict.scope})")


if __name__ == "__main__":
    ns = parse_args()
    logging.basicConfig(
        level=logging.INFO if ns.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(ns)
