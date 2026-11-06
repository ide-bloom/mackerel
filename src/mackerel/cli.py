"""Command line interface: ``mackerel <command>``.

Commands are added milestone by milestone; every command is deterministic (seeded) so the
figures in the documentation can be regenerated with one call.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from mackerel import __version__

VALIDATION_SCRIPTS = {
    "foundations": "validation.foundations",
    "advection": "validation.advection",
    "projection": "validation.projection",
    "taylor_green": "validation.taylor_green",
    "poiseuille": "validation.poiseuille",
    "lid_cavity": "validation.lid_cavity",
    "cylinder": "validation.cylinder",
    "smoke": "validation.smoke",
    "moving": "validation.moving",
    "flip": "validation.flip",
    "performance": "validation.performance",
    "webtest": "validation.web_reference",
    "figures": "validation.figures",
}

SCENES = {
    "smoke": "mackerel.scenes:smoke_plume",
    "jet": "mackerel.scenes:jet",
    "cavity": "mackerel.scenes:cavity",
    "cylinder": "mackerel.scenes:cylinder",
    "moving": "mackerel.scenes:moving",
    "dam_break": "mackerel.scenes:dam_break",
    "splash": "mackerel.scenes:splash",
    "sloshing": "mackerel.scenes:sloshing",
    "taylor_green": "mackerel.scenes:taylor_green",
}


def _run_validation(name: str, extra: list[str]) -> int:
    module_name = VALIDATION_SCRIPTS[name]
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        print(f"{module_name} has no main()", file=sys.stderr)
        return 2
    result: int = main(extra)
    return result


def _run_scene(name: str, extra: list[str]) -> int:
    from mackerel.scenes import run_scene

    result: int = run_scene(name, extra)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mackerel", description=__doc__)
    p.add_argument("--version", action="version", version=f"mackerel {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print version and environment information")

    v = sub.add_parser("validate", help="run a validation / figure script")
    v.add_argument("name", choices=sorted(VALIDATION_SCRIPTS))
    v.add_argument("args", nargs=argparse.REMAINDER)

    s = sub.add_parser("run", help="run a demo scene and write output frames")
    s.add_argument("name", choices=sorted(SCENES))
    s.add_argument("args", nargs=argparse.REMAINDER)

    b = sub.add_parser("benchmark", help="run the performance benchmarks")
    b.add_argument("outdir", nargs="?", default=None)
    b.add_argument("--repeats", type=int, default=7)
    b.add_argument("--steps", type=int, default=40)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        from mackerel.diagnostics import ffmpeg_report

        print(f"mackerel {__version__}")
        print(ffmpeg_report())
        return 0
    if args.command == "validate":
        return _run_validation(args.name, list(args.args))
    if args.command == "run":
        return _run_scene(args.name, list(args.args))
    if args.command == "benchmark":
        module = importlib.import_module("validation.performance")
        benchmark_args = ([] if args.outdir is None else [args.outdir])
        benchmark_args += ["--repeats", str(args.repeats), "--steps", str(args.steps)]
        result: int = module.main(benchmark_args)
        return result
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
