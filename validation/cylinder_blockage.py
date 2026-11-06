"""Task 3: measured cylinder Strouhal response to blockage and grid resolution.

The M6 baseline is a 2x1 channel with D/H=.16.  This driver uses a longer 4x1 channel for
new cases, keeps U=1 and Re=U D/nu=100, and measures the same downstream transverse-velocity
signal with the existing peak/FFT estimator.  The .08 case is run at two grid spacings;
the .04 case is deliberately reported with its cells-per-diameter limitation.

Run ``python -m validation.cylinder_blockage``; outputs go to
``outputs/m6/blockage``.  ``--case`` and ``--end-time`` are available for reproducible
subsets, not for selecting preferred results.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme
from mackerel.diagnostics import vorticity_field
from mackerel.grid import Grid, apply_velocity_bc
from mackerel.obstacles import apply_solid_velocity, circle_mask
from mackerel.simulate import Simulation, StepInfo
from validation.cylinder import channel_bc, strouhal_from_signal

OUT = Path("outputs/m6/blockage")


def run_case(
    name: str,
    *,
    nx: int,
    ny: int,
    lx: float,
    diameter: float,
    end_time: float = 20.0,
    dt: float = 0.005,
    reynolds: float = 100.0,
) -> dict[str, Any]:
    """Run one geometry with square cells and return measured frequency diagnostics."""
    grid = Grid(nx, ny, lx=lx)
    height = grid.ly
    if not np.isclose(height, 1.0):
        raise ValueError("this comparison uses H=1; choose nx/lx and ny accordingly")
    speed = 1.0
    nu = speed * diameter / reynolds
    center = (0.5, 0.5)
    solid = circle_mask(grid, center, diameter / 2.0)
    sim = Simulation.at_rest(
        grid, channel_bc(speed), nu=nu, dt=dt, scheme=Scheme.MACCORMACK,
        pressure_tol=1e-8, pressure_maxiter=2000,
    )
    sim.set_solid(solid)
    sim.u[:] = speed
    # Deterministic, geometry-scaled asymmetry selects a shedding phase without forcing a
    # frequency.  Its amplitude is identical for every case.
    X, Y = grid.cell_coords()
    perturbation = 0.01 * np.exp(
        -((X - (center[0] + 2.0 * diameter)) ** 2
          + (Y - (center[1] + 0.25 * diameter)) ** 2) / max(diameter**2, 1e-12)
    )
    sim.v[1:-1, :] += 0.5 * (perturbation[:-1, :] + perturbation[1:, :])
    apply_velocity_bc(sim.u, sim.v, sim.bc)
    apply_solid_velocity(sim.u, sim.v, solid)
    probe_x, probe_y = 1.5, 0.5
    pi = min(grid.nx - 1, int(probe_x / grid.h))
    pj = min(grid.ny - 1, int(probe_y / grid.h))
    times: list[float] = []
    probe: list[float] = []

    def record(state: Simulation, _info: StepInfo) -> None:
        if not times or state.time - times[-1] >= 0.049999:
            times.append(state.time)
            probe.append(float(grid.v_to_centers(state.v)[pj, pi]))

    start = time.perf_counter()
    sim.run_until(end_time, record)
    measurement = strouhal_from_signal(
        np.asarray(times), np.asarray(probe), diameter, speed, start=0.35 * end_time
    )
    return {
        "name": name,
        "nx": nx,
        "ny": ny,
        "domain_length": lx,
        "domain_height": height,
        "diameter": diameter,
        "blockage": diameter / height,
        "cells_per_diameter": diameter / grid.h,
        "reynolds": reynolds,
        "nu": nu,
        "dt": dt,
        "time": sim.time,
        "steps": len(sim.history),
        "wall_time": time.perf_counter() - start,
        "reference_strouhal": float(-3.3265 / reynolds + 0.1816 + 0.00016 * reynolds),
        "final_divergence": sim.history[-1].div_max,
        "max_divergence": max(item.div_max for item in sim.history),
        "energy": sim.energy(),
        "probe_x": probe_x,
        "probe_y": probe_y,
        "times": np.asarray(times),
        "probe": np.asarray(probe),
        "vorticity": vorticity_field(sim.u, sim.v, grid, sim.bc),
        "solid": solid,
        **measurement,
    }


def _serial(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--case", choices=["08_coarse", "08_fine", "04", "04_fine"], default=None)
    parser.add_argument("--end-time", type=float, default=None)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    definitions: dict[str, dict[str, Any]] = {
        "08_coarse": {"nx": 128, "ny": 32, "lx": 4.0, "diameter": 0.08},
        "08_fine": {"nx": 256, "ny": 64, "lx": 4.0, "diameter": 0.08},
        "04": {"nx": 256, "ny": 64, "lx": 4.0, "diameter": 0.04},
        "04_fine": {"nx": 320, "ny": 80, "lx": 4.0, "diameter": 0.04},
    }
    selected = list(definitions) if args.case is None else [args.case]
    rows_by_name: dict[str, dict[str, Any]] = {}
    for path in args.outdir.glob("*.json"):
        if path.name == "blockage.json":
            continue
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "name" in value:
            rows_by_name[value["name"]] = value
    for name in selected:
        print(f"running {name}", flush=True)
        default_end = 12.0 if name == "04_fine" else 20.0
        end_time = default_end if args.end_time is None else args.end_time
        row = run_case(name, end_time=end_time, **definitions[name])
        np.savez(
            args.outdir / f"{name}.npz", times=row["times"], probe=row["probe"],
            vorticity=row["vorticity"], solid=row["solid"],
        )
        # Save after every case so a long parameter study cannot lose completed measurements.
        serial_row = _serial(row)
        (args.outdir / f"{name}.json").write_text(json.dumps(serial_row, indent=2) + "\n")
        rows_by_name[name] = serial_row
        serial = [rows_by_name[key] for key in definitions if key in rows_by_name]
        (args.outdir / "blockage.json").write_text(json.dumps(serial, indent=2) + "\n")
        with (args.outdir / "blockage.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(serial[0]))
            writer.writeheader()
            writer.writerows(serial)
        print(json.dumps(_serial(row), indent=2), flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    rows = [rows_by_name[key] for key in definitions if key in rows_by_name]
    for row in rows:
        st = row["strouhal"]
        if st is not None:
            ax.scatter(row["blockage"], st, s=55, label=f"{row['name']} ({row['cells_per_diameter']:.2f} cells/D)")
    reference = rows[0]["reference_strouhal"]
    ax.axhline(reference, color="black", ls="--", label=f"reference relation {reference:.6f}")
    ax.set(xlabel="blockage D/H", ylabel="peak-to-peak Strouhal St", title="Cylinder blockage study")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(args.outdir / "blockage.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
