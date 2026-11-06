"""Two-resolution dam-break comparison with an explicit gate-time convention.

Both PIC and 95% FLIP are run at 48x36 and 96x72.  The initial column width is a=.25,
height=2a, gravity=9.81, and tau=t*sqrt(2g/a).  Time zero is instantaneous release;
GATE_OPENING_DELAY=0 means no +.175 (or other) shift is applied to either curve.  The front
is the rightmost particle plus half a cell, as in the original M7 protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.flip import (
    GATE_OPENING_DELAY,
    _save_result_json,
    _write_series,
    load_reference,
    run_dam_break,
)

OUT = Path("outputs/m7/resolution")


def _serial(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {"records", "frames", "particles", "water", "series"}
    return {
        key: value for key, value in result.items()
        if key not in excluded and not isinstance(value, np.ndarray)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--n", type=int, choices=[48, 96], default=None)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    resolutions = [48, 96] if args.n is None else [args.n]
    rows: dict[str, dict[str, Any]] = {}
    for path in args.outdir.glob("case_*.json"):
        value = json.loads(path.read_text())
        rows[value["case"]] = value
    for n in resolutions:
        for blend, method in ((0.05, "flip"), (1.0, "pic")):
            name = f"{method}_{n}"
            print(f"running {name}", flush=True)
            result = run_dam_break(n=n, pic_blend=blend)
            row = _serial(result)
            row["case"] = name
            rows[name] = row
            _save_result_json(args.outdir / f"case_{name}.json", result)
            # Add the case label after the common serializer has written the measured fields.
            (args.outdir / f"case_{name}.json").write_text(json.dumps(row, indent=2) + "\n")
            _write_series(args.outdir / f"case_{name}.csv", result["records"])
            arrays: dict[str, Any] = {
                "reference_tau": result["reference_tau"],
                "reference_delta": result["reference_delta"],
                "predicted_reference_delta": result["predicted_reference_delta"],
                "records": np.asarray([[r["time"], r["tau"], r["delta"], r["volume"]] for r in result["records"]]),
            }
            p = result["particles"]
            arrays.update(x=p.x, y=p.y, u=p.u, v=p.v)
            np.savez(args.outdir / f"case_{name}.npz", **arrays)
            combined = [rows[key] for key in sorted(rows)]
            report = {
                "gate_opening_delay": GATE_OPENING_DELAY,
                "time_convention": "instantaneous release; tau=(t+0)*sqrt(2*g/a); no reference-time shift",
                "front_convention": "rightmost particle plus half a cell, divided by a",
                "cases": combined,
            }
            (args.outdir / "dam_break_resolution.json").write_text(json.dumps(report, indent=2) + "\n")
            with (args.outdir / "dam_break_resolution.csv").open("w", newline="") as f:
                fields = ["case", "n", "pic_blend", "time", "steps", "front_rmse",
                          "front_max_abs_error", "front_mean_relative_error",
                          "volume_relative_error", "max_divergence", "wall_time"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows([{key: row[key] for key in fields} for row in combined])
            print(json.dumps({k: row[k] for k in fields}, indent=2), flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_tau, ref_delta = load_reference()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for n in (48, 96):
        for method, color in (("flip", "tab:blue"), ("pic", "tab:orange")):
            name = f"{method}_{n}"
            if name not in rows:
                continue
            data = np.load(args.outdir / f"case_{name}.npz")["records"]
            axes[0].plot(data[:, 1], data[:, 2], color=color, alpha=0.45 if n == 48 else 1.0,
                         label=f"{method.upper()} {n}")
    axes[0].plot(ref_tau, ref_delta, "ko", ms=4, label="Martin-Moyce")
    axes[0].set(xlabel="tau (zero gate delay)", ylabel="delta", title="Dam-break front refinement")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    names = [name for name in sorted(rows) if name.rsplit("_", 1)[-1].isdigit()]
    axes[1].bar(names, [rows[name]["front_rmse"] for name in names], color=["tab:blue" if "flip" in n else "tab:orange" for n in names])
    axes[1].set(ylabel="front RMSE", title="Error at the 14 reference stations")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(args.outdir / "dam_break_resolution.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
