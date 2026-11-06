"""Task 4: Re=1000 and fully budgeted 128² Re=400 cavity comparisons.

The reference values are the unchanged Ghia--Ghia--Shin tables.  The default cases are
Re=1000 on 64² and Re=400 on 128², both with dt=.01 and a 2000-iteration pressure cap.
Each case is saved separately so a long run can be reproduced without rerunning the other.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.lid_cavity import run_cavity

OUT = Path("outputs/m5/extended")


def _serial(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--case", choices=["re1000_64", "re400_128"], default=None)
    parser.add_argument("--max-time", type=float, default=None)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    definitions: dict[str, dict[str, Any]] = {
        "re1000_64": {"reynolds": 1000, "n": 64, "max_time": 200.0},
        "re400_128": {"reynolds": 400, "n": 128, "max_time": 100.0},
    }
    selected = list(definitions) if args.case is None else [args.case]
    rows: dict[str, dict[str, Any]] = {}
    for path in args.outdir.glob("*.json"):
        if path.name == "cavity_extended.json":
            continue
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "case" in value:
            rows[value["case"]] = value
    for name in selected:
        params = dict(definitions[name])
        if args.max_time is not None:
            params["max_time"] = args.max_time
        print(f"running {name}: {params}", flush=True)
        row = run_cavity(**params)
        row["case"] = name
        serial = _serial(row)
        rows[name] = serial
        (args.outdir / f"{name}.json").write_text(json.dumps(serial, indent=2) + "\n")
        arrays: dict[str, Any] = {
            key: value for key, value in row.items() if isinstance(value, np.ndarray)
        }
        np.savez(args.outdir / f"{name}.npz", **arrays)
        combined = [rows[key] for key in definitions if key in rows]
        (args.outdir / "cavity_extended.json").write_text(json.dumps(combined, indent=2) + "\n")
        with (args.outdir / "cavity_extended.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
        print(json.dumps(serial, indent=2), flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = [rows[key] for key in definitions if key in rows]
    fig, axes = plt.subplots(len(ordered), 2, figsize=(11, 4.2 * len(ordered)), squeeze=False, layout="constrained")
    for r, row in enumerate(ordered):
        data = np.load(args.outdir / f"{row['case']}.npz")
        axes[r, 0].plot(data["u_line"], data["y_line"], "b-", label="mackerel")
        axes[r, 0].plot(data["u_reference"], data["y_reference"], "ro", ms=3, label="Ghia")
        axes[r, 1].plot(data["x_line"], data["v_line"], "b-", label="mackerel")
        axes[r, 1].plot(data["x_reference"], data["v_reference"], "ro", ms=3, label="Ghia")
        axes[r, 0].set_title(f"{row['case']}: u max/RMS {row['u_max_error']:.4f}/{row['u_rms_error']:.4f}")
        axes[r, 1].set_title(f"v primary/all max {row['v_max_error']:.4f}/{row['v_max_error_all']:.4f}")
        for ax in axes[r]:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.savefig(args.outdir / "cavity_extended.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
