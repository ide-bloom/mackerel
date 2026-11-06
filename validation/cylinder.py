"""Milestone 6: cylinder wake, Karman street and Strouhal measurement.

A unit-speed stream enters the left side of a 2x1 channel and leaves through a zero-gradient
outflow.  A staircase circular obstacle is held at rest.  At Re=100 the wake becomes
unsteady; a downstream transverse-velocity probe supplies a measured shedding frequency,
``St=fD/U``.  No frequency is imposed by the driver.

Run ``python -m validation.cylinder``; outputs are written to ``outputs/m6``.  The default
64x32 grid and 30-time-unit run are deliberately reproducible rather than a claim of a
body-fitted, grid-independent cylinder calculation.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from mackerel.advect import Scheme
from mackerel.diagnostics import vorticity_field
from mackerel.grid import BoundaryConditions, Grid, Kind, SideBC, apply_velocity_bc
from mackerel.obstacles import apply_solid_velocity, circle_mask
from mackerel.simulate import Simulation, StepInfo

OUT = Path("outputs/m6")


def channel_bc(speed: float = 1.0) -> BoundaryConditions:
    return BoundaryConditions(
        SideBC(Kind.INFLOW, velocity=(speed, 0.0)),
        SideBC(Kind.OUTFLOW),
        SideBC(Kind.NO_SLIP),
        SideBC(Kind.NO_SLIP),
    )


def strouhal_from_signal(
    times: np.ndarray, signal: np.ndarray, diameter: float, speed: float, start: float
) -> dict[str, Any]:
    """Measure peak-to-peak and FFT frequencies after the transient."""
    selected = times >= start
    t = np.asarray(times[selected], dtype=np.float64)
    y = np.asarray(signal[selected], dtype=np.float64)
    if len(t) < 8:
        return {"frequency": None, "strouhal": None, "fft_frequency": None, "peaks": []}
    peaks, _ = find_peaks(y, prominence=max(1e-4, 0.05 * float(np.std(y))))
    peak_times = t[peaks]
    frequency: float | None = None
    if len(peak_times) >= 2:
        frequency = float(1.0 / np.mean(np.diff(peak_times)))
    dt = float(np.mean(np.diff(t)))
    spectrum = np.abs(np.fft.rfft(y - np.mean(y)))
    frequencies = np.fft.rfftfreq(len(y), d=dt)
    spectrum[0] = 0.0
    fft_frequency = float(frequencies[int(np.argmax(spectrum))])
    return {
        "frequency": frequency,
        "strouhal": None if frequency is None else frequency * diameter / speed,
        "fft_frequency": fft_frequency,
        "fft_strouhal": fft_frequency * diameter / speed,
        "peaks": peak_times,
        "amplitude_std": float(np.std(y)),
        "sample_count": len(t),
    }


def run_cylinder(
    n: int = 32,
    reynolds: float = 100.0,
    end_time: float = 30.0,
    dt: float = 0.005,
) -> dict[str, Any]:
    """Run the wake and return fields, probe data and measured frequency information."""
    grid = Grid(2 * n, n, lx=2.0)
    diameter = 0.16
    speed = 1.0
    nu = speed * diameter / reynolds
    center = (0.35, 0.5)
    solid = circle_mask(grid, center, diameter / 2.0)
    sim = Simulation.at_rest(
        grid,
        channel_bc(speed),
        nu=nu,
        dt=dt,
        scheme=Scheme.MACCORMACK,
        pressure_tol=1e-8,
        pressure_maxiter=2000,
    )
    sim.set_solid(solid)
    sim.u[:] = speed
    # A tiny deterministic asymmetric perturbation selects a shedding phase; it is not a
    # periodic forcing and decays if the Reynolds number is below the shedding threshold.
    X, Y = grid.cell_coords()
    perturbation = 0.01 * np.exp(-((X - 0.5) ** 2 + (Y - 0.52) ** 2) / 0.01)
    sim.v[1:-1, :] += 0.5 * (perturbation[:-1, :] + perturbation[1:, :])
    apply_velocity_bc(sim.u, sim.v, sim.bc)
    apply_solid_velocity(sim.u, sim.v, solid)

    probe_x, probe_y = 0.8, 0.5
    pi = min(grid.nx - 1, int(probe_x / grid.h))
    pj = min(grid.ny - 1, int(probe_y / grid.h))
    times: list[float] = []
    probe: list[float] = []

    def record(state: Simulation, _info: StepInfo) -> None:
        if len(times) == 0 or state.time - times[-1] >= 0.049999:
            times.append(state.time)
            probe.append(float(grid.v_to_centers(state.v)[pj, pi]))

    t0 = time.perf_counter()
    sim.run_until(end_time, record)
    measurement = strouhal_from_signal(
        np.asarray(times), np.asarray(probe), diameter, speed, start=0.35 * end_time
    )
    return {
        "n": n,
        "reynolds": reynolds,
        "diameter": diameter,
        "speed": speed,
        "nu": nu,
        "dt": dt,
        "time": float(sim.time),
        "steps": len(sim.history),
        "wall_time": time.perf_counter() - t0,
        "probe_x": probe_x,
        "probe_y": probe_y,
        "final_divergence": float(sim.history[-1].div_max),
        "max_divergence": float(max(d.div_max for d in sim.history)),
        "energy": sim.energy(),
        "reference_strouhal": float(-3.3265 / reynolds + 0.1816 + 0.00016 * reynolds),
        **measurement,
        "times": np.asarray(times),
        "probe": np.asarray(probe),
        "u": sim.u,
        "v": sim.v,
        "solid": solid,
        "vorticity": vorticity_field(sim.u, sim.v, grid, sim.bc),
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_cylinder()
    scalar = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    (outdir / "cylinder.json").write_text(json.dumps(scalar, indent=2) + "\n")
    with (outdir / "cylinder.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar))
        writer.writeheader()
        writer.writerow(scalar)
    arrays: dict[str, Any] = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez(outdir / "cylinder_fields.npz", **arrays)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    ax = axes[0]
    vort = np.asarray(result["vorticity"])
    # corners -> cell-centred average just for the raster visualisation
    vc = 0.25 * (vort[:-1, :-1] + vort[1:, :-1] + vort[:-1, 1:] + vort[1:, 1:])
    vc = np.where(result["solid"], np.nan, vc)
    im = ax.imshow(vc, origin="lower", extent=(0, 2, 0, 1), cmap="RdBu_r",
                   vmin=-8, vmax=8, aspect="auto")
    ax.add_patch(Circle((0.35, 0.5), 0.08, color="black", fill=True))
    ax.plot([0.8], [0.5], "kx", ms=8, label="velocity probe")
    ax.set_title(f"Cylinder vorticity, Re={result['reynolds']:.0f}, t={result['time']:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="vorticity")
    ax = axes[1]
    ax.plot(result["times"], result["probe"], "b-", lw=1)
    ax.axvline(0.35 * result["time"], color="k", ls=":", label="frequency window")
    ax.set_xlabel("time")
    ax.set_ylabel("v at downstream probe")
    ax.set_title(f"St={result['strouhal']} (peak-to-peak), FFT St={result['fft_strouhal']:.4f}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "cylinder_wake.png", dpi=150)
    plt.close(fig)

    print("Cylinder wake validation")
    for key in ("reynolds", "n", "time", "steps", "wall_time", "final_divergence"):
        print(f"  {key}: {result[key]}")
    print(
        f"  St peak-to-peak={result['strouhal']}  St FFT={result['fft_strouhal']:.5f} "
        f"reference={result['reference_strouhal']:.5f}"
    )
    print(f"figures and data written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
