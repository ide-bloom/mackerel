"""Strouhal number as a function of blockage and of cells-per-diameter.

The M6 blockage study reported three values that disagree with the reference relation in the
opposite direction from a pure blockage trend: St=.204 at D/H=.16, St=.153 at D/H=.08
(5.12 cells/D) and St=.121 at D/H=.04 (3.2 cells/D), against a reference of about .164.
Because cells-per-diameter and blockage were varied together, that table cannot say whether
the gap at low blockage is a resolution artefact or real confinement physics.

This driver separates the two axes with a fixed protocol.

* **Study A (constant cells/D=8, varying D/H).**  Space and time per diameter are held fixed
  (``D/dt = 16``, ``u dt/h = 0.5``), so the only variable is blockage.  Any trend here is the
  confinement response of the numerical solution at a fixed numerical quality.
* **Study B (constant D/H=.16, varying cells/D).**  ``dt`` is held fixed, so only the grid is
  refined; the trend is the spatial discretisation error.

Both studies share one geometry and one signal: a 4x1 channel, a staircase circle of diameter
D centred at (0.5, 0.5), U=1 inflow, Re=UD/nu=100, and the transverse velocity at
(1.5, 0.5) sampled every 0.025 time units.  The shedding frequency is measured by a Hann
window over the second half of the run; the window length, the number of shedding cycles it
contains and the frequency resolution (1/window) are reported with every St so the estimate
can be judged rather than taken on trust.

Run ``python -m validation.cylinder_resolution --study a`` (or ``b``, ``control``); each
completed case is written to ``outputs/m7/strouhal/<case>.json`` before the next one starts,
so an interrupted study never loses measurements and can be resumed.
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
from mackerel.grid import Grid, apply_velocity_bc
from mackerel.obstacles import apply_solid_velocity, circle_mask
from mackerel.simulate import Simulation, StepInfo
from validation.cylinder import channel_bc, strouhal_from_signal

OUT = Path("outputs/m7/strouhal")

#: The free-stream reference relation used by the M6 blockage study, at Re=100.
REFERENCE_REYNOLDS = 100.0

#: Probe used by every case in this study.
PROBE = (1.5, 0.5)


def _peak_frequency(y: np.ndarray, dt: float, pad_factor: int = 8) -> float | None:
    """Dominant frequency of an already-unit-spaced, detrended signal."""
    if y.size < 16 or not np.any(y):
        return None
    window = np.hanning(y.size)
    n = int(pad_factor * y.size)
    spectrum = np.abs(np.fft.rfft(y * window, n))
    spectrum[0] = 0.0
    k = int(np.argmax(spectrum))
    freqs = np.fft.rfftfreq(n, d=dt)
    offset = 0.0
    if 0 < k < spectrum.size - 1:
        # Parabolic interpolation of the log magnitude: below the window's own resolution the
        # peak position is still meaningful, but only to a fraction of a bin.
        e = 1e-300
        a, b, c = (
            float(np.log(spectrum[k - 1] + e)),
            float(np.log(spectrum[k] + e)),
            float(np.log(spectrum[k + 1] + e)),
        )
        denominator = a - 2.0 * b + c
        if denominator != 0.0:
            offset = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
    return float(freqs[k] + offset * (freqs[1] - freqs[0]))


def windowed_strouhal(
    times: np.ndarray,
    signal: np.ndarray,
    diameter: float,
    speed: float,
    start: float,
    pad_factor: int = 8,
) -> dict[str, Any]:
    """Strouhal number from a Hann-windowed spectrum of the signal after ``start``.

    The window is ``[start, times[-1]]``.  A least-squares straight line is removed before
    windowing so a residual transient drift cannot leak into the lowest bins.  The reported
    ``frequency_resolution`` is ``1/window``; a Hann window spreads a single tone over roughly
    twice that, so the uncertainty on St is at least ``2 diameter / (speed * window)``.
    The signal is cut in half inside the window and both halves are measured separately, which
    gives an empirical spread for the same peak.
    """
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    selected = t >= start
    t = t[selected]
    y = y[selected]
    empty: dict[str, Any] = {
        "frequency": None,
        "strouhal": None,
        "window_start": float(start),
        "window": None,
        "cycles": None,
        "frequency_resolution": None,
        "strouhal_half_first": None,
        "strouhal_half_second": None,
        "strouhal_half_spread": None,
        "amplitude_std": None,
        "sample_count": int(t.size),
    }
    if t.size < 32:
        return empty
    dt = float(np.median(np.diff(t)))
    uniform = np.arange(t[0], t[-1], dt)
    if uniform.size < 32:
        return empty
    resampled = np.interp(uniform, t, y)
    origin = uniform - uniform[0]
    design = np.vstack([origin, np.ones_like(origin)]).T
    coefficients, *_ = np.linalg.lstsq(design, resampled, rcond=None)
    detrended = resampled - design @ coefficients
    window = float(uniform[-1] - uniform[0])
    frequency = _peak_frequency(detrended, dt, pad_factor)
    if frequency is None or frequency <= 0.0:
        return empty
    half = detrended.size // 2
    first = _peak_frequency(detrended[:half], dt, pad_factor)
    second = _peak_frequency(detrended[half:], dt, pad_factor)
    scale = float(diameter) / float(speed)
    strouhal = frequency * scale
    first_st = None if first is None else first * scale
    second_st = None if second is None else second * scale
    spread = (
        None
        if first_st is None or second_st is None
        else 0.5 * abs(first_st - second_st)
    )
    return {
        "frequency": frequency,
        "strouhal": strouhal,
        "window_start": float(uniform[0]),
        "window": window,
        "cycles": frequency * window,
        "frequency_resolution": 1.0 / window,
        "strouhal_half_first": first_st,
        "strouhal_half_second": second_st,
        "strouhal_half_spread": spread,
        "amplitude_std": float(np.std(detrended)),
        "sample_count": int(t.size),
    }


def run_case(
    name: str,
    *,
    nx: int,
    ny: int,
    lx: float,
    diameter: float,
    end_time: float,
    dt: float,
    reynolds: float = REFERENCE_REYNOLDS,
    sample_interval: float = 0.025,
    checkpoint: Path | None = None,
    checkpoint_every: float = 1.0,
) -> dict[str, Any]:
    """Run one cylinder case with square cells and measure its shedding frequency.

    If ``checkpoint`` is given, the velocity field, clock and probe history are written every
    ``checkpoint_every`` time units and the file is deleted when the case completes.  A case
    interrupted mid-run (a wall-clock limit, a restarted machine) therefore resumes instead of
    starting over, which is what makes the multi-hour cells/D=8 and cells/D=16 cases feasible.
    """
    grid = Grid(nx, ny, lx=lx)
    height = grid.ly
    if not np.isclose(height, 1.0):
        raise ValueError("this comparison uses H=1; choose nx/lx and ny accordingly")
    speed = 1.0
    nu = speed * diameter / reynolds
    center = (0.5, 0.5)
    solid = circle_mask(grid, center, diameter / 2.0)
    sim = Simulation.at_rest(
        grid,
        channel_bc(speed),
        nu=nu,
        dt=dt,
        scheme=Scheme.MACCORMACK,
        # The linear solve is the whole cost at 200x50 and above, so the same tolerance as the
        # M6 blockage driver is kept; the measured peak iteration count is recorded with each
        # case and any cap being hit is reported explicitly.
        pressure_tol=1e-8,
        pressure_maxiter=2000,
    )
    sim.set_solid(solid)
    sim.u[:] = speed
    resumed = False
    resumed_from = 0.0
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as saved:
            sim.u = np.asarray(saved["u"], dtype=np.float64)
            sim.v = np.asarray(saved["v"], dtype=np.float64)
            sim.time = float(saved["time"])
            saved_times = list(np.asarray(saved["times"], dtype=np.float64))
            saved_probe = list(np.asarray(saved["probe"], dtype=np.float64))
        resumed = True
        resumed_from = sim.time
        times: list[float] = saved_times
        probe: list[float] = saved_probe
        print(f"  resuming from t={sim.time:.3f}", flush=True)
    else:
        times = []
        probe = []
        # Identical to the M6 blockage driver: a deterministic, geometry-scaled asymmetry picks
        # a shedding phase.  Its amplitude is the same for every case, it forces no frequency.
        X, Y = grid.cell_coords()
        perturbation = 0.01 * np.exp(
            -(
                (X - (center[0] + 2.0 * diameter)) ** 2
                + (Y - (center[1] + 0.25 * diameter)) ** 2
            )
            / max(diameter**2, 1e-12)
        )
        sim.v[1:-1, :] += 0.5 * (perturbation[:-1, :] + perturbation[1:, :])
        apply_velocity_bc(sim.u, sim.v, sim.bc)
        apply_solid_velocity(sim.u, sim.v, solid)
    probe_x, probe_y = PROBE
    pi = min(grid.nx - 1, int(probe_x / grid.h))
    pj = min(grid.ny - 1, int(probe_y / grid.h))
    last_checkpoint = [sim.time]

    def record(state: Simulation, _info: StepInfo) -> None:
        if not times or state.time - times[-1] >= sample_interval - 1e-9:
            times.append(state.time)
            probe.append(float(grid.v_to_centers(state.v)[pj, pi]))
        if checkpoint is not None and state.time - last_checkpoint[0] >= checkpoint_every:
            last_checkpoint[0] = state.time
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                checkpoint,
                u=state.u,
                v=state.v,
                time=np.float64(state.time),
                times=np.asarray(times),
                probe=np.asarray(probe),
            )

    started = time.perf_counter()
    infos = sim.run_until(end_time, record)
    wall = time.perf_counter() - started
    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()
    t = np.asarray(times)
    y = np.asarray(probe)
    start = 0.5 * end_time
    measurement = windowed_strouhal(t, y, diameter, speed, start)
    baseline = strouhal_from_signal(t, y, diameter, speed, start)
    return {
        "name": name,
        "nx": nx,
        "ny": ny,
        "domain_length": lx,
        "domain_height": height,
        "spacing": grid.h,
        "diameter": diameter,
        "blockage": diameter / height,
        "cells_per_diameter": diameter / grid.h,
        "reynolds": reynolds,
        "nu": nu,
        "dt": dt,
        "diameter_per_dt": diameter / dt,
        "cfl_advective": speed * dt / grid.h,
        "end_time": end_time,
        "time": sim.time,
        "steps": len(infos),
        "wall_time": wall,
        "resumed": resumed,
        "resumed_from_time": resumed_from,
        "reference_strouhal": float(
            -3.3265 / reynolds + 0.1816 + 0.00016 * reynolds
        ),
        "final_divergence": sim.history[-1].div_max,
        "max_divergence": max(item.div_max for item in sim.history),
        "max_pressure_iterations": max(item.pressure_iterations for item in infos),
        "max_pressure_residual": max(item.pressure_residual for item in infos),
        "pressure_cap_hits": sum(1 for item in infos if item.pressure_iterations >= 2000),
        "energy": sim.energy(),
        "probe_x": probe_x,
        "probe_y": probe_y,
        "times": t,
        "probe": y,
        "legacy_peak_strouhal": baseline["strouhal"],
        "legacy_peak_count": len(baseline["peaks"]),
        **measurement,
    }


#: Study A: cells/D held at 8, blockage varied.  dt follows D so D/dt=16 and u dt/h=0.5.
#: Ordered cheapest first, so an interrupted study has already banked the most cases.
STUDY_A: dict[str, dict[str, Any]] = {
    "A_b032_c8": {"nx": 100, "ny": 25, "lx": 4.0, "diameter": 0.32, "end_time": 40.0, "dt": 0.02},
    "A_b020_c8": {"nx": 160, "ny": 40, "lx": 4.0, "diameter": 0.20, "end_time": 30.0, "dt": 0.0125},
    "A_b016_c8": {"nx": 200, "ny": 50, "lx": 4.0, "diameter": 0.16, "end_time": 30.0, "dt": 0.01},
    "A_b008_c8": {"nx": 400, "ny": 100, "lx": 4.0, "diameter": 0.08, "end_time": 30.0, "dt": 0.005},
}

#: Study B: D/H held at .16, grid refined at fixed dt (D/dt=32 for the 6/12 cases, D/dt=25 for
#: c8 so that its half-dt control below halves an exact number).  cells/D = 6, 8, 12 is the
#: requested ladder inside the affordable band; 16 is attempted and its cost stated if it does
#: not finish (measured: 0.67 s per step at 400x100, i.e. about 6.7 h per 30-time-unit case).
STUDY_B: dict[str, dict[str, Any]] = {
    "B_c6": {"nx": 152, "ny": 38, "lx": 4.0, "diameter": 0.16, "end_time": 25.0, "dt": 0.005},
    "B_c12": {"nx": 300, "ny": 75, "lx": 4.0, "diameter": 0.16, "end_time": 25.0, "dt": 0.005},
    "B_c8": {"nx": 200, "ny": 50, "lx": 4.0, "diameter": 0.16, "end_time": 30.0, "dt": 0.005},
    "B_c16": {"nx": 400, "ny": 100, "lx": 4.0, "diameter": 0.16, "end_time": 30.0, "dt": 0.005},
}

#: Is the time step converged where the resolution study sits?  Same grid, one halving of dt
#: in each direction around dt=0.005 at cells/D=8.
CONTROL: dict[str, dict[str, Any]] = {
    "C_c8_dt01": {"nx": 200, "ny": 50, "lx": 4.0, "diameter": 0.16, "end_time": 30.0, "dt": 0.01},
    "C_c8_dt005": {"nx": 200, "ny": 50, "lx": 4.0, "diameter": 0.16, "end_time": 30.0, "dt": 0.005},
}

STUDIES: dict[str, dict[str, dict[str, Any]]] = {
    "a": STUDY_A,
    "b": STUDY_B,
    "control": CONTROL,
}


def _serial(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}


def _write_study(outdir: Path, study: str, definitions: dict[str, dict[str, Any]]) -> None:
    rows = []
    for key in definitions:
        path = outdir / f"{key}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if not rows:
        return
    rows.sort(key=lambda row: (row["blockage"], row["cells_per_diameter"]))
    (outdir / f"study_{study}.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "name",
        "blockage",
        "cells_per_diameter",
        "spacing",
        "dt",
        "diameter_per_dt",
        "strouhal",
        "cycles",
        "window",
        "frequency_resolution",
        "strouhal_half_first",
        "strouhal_half_second",
        "strouhal_half_spread",
        "legacy_peak_strouhal",
        "legacy_peak_count",
        "reference_strouhal",
        "amplitude_std",
    ]
    with (outdir / f"study_{study}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(outdir: Path, study: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((outdir / f"study_{study}.json").read_text())
    figure, axes = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    reference = rows[0]["reference_strouhal"]
    axes.axhline(reference, color="black", ls="--", label=f"reference relation {reference:.4f}")
    if study == "a":
        for row in rows:
            if row["strouhal"] is None:
                continue
            axes.errorbar(
                row["blockage"],
                row["strouhal"],
                yerr=row["strouhal_half_spread"],
                fmt="o",
                capsize=3,
                label=f"{row['name']} ({row['cells_per_diameter']:.2f} cells/D)",
            )
        axes.set(
            xlabel="blockage D/H",
            ylabel="windowed Strouhal St",
            title="Study A: constant 8 cells/D, varying blockage",
        )
    else:
        for row in rows:
            if row["strouhal"] is None:
                continue
            axes.errorbar(
                row["cells_per_diameter"],
                row["strouhal"],
                yerr=row["strouhal_half_spread"],
                fmt="o",
                capsize=3,
                label=f"{row['name']} (D/dt={row['diameter_per_dt']:.0f})",
            )
        axes.set(
            xlabel="cells per diameter",
            ylabel="windowed Strouhal St",
            title=f"Study {study.upper()}: D/H=.16, varying grid",
        )
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8)
    figure.savefig(outdir / f"study_{study}.png", dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--study", choices=sorted(STUDIES), required=True)
    parser.add_argument("--case", nargs="*", default=None, help="subset of case names")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run cases whose JSON already exists (the default resumes a study)",
    )
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    definitions = STUDIES[args.study]
    selected = list(definitions) if args.case is None else args.case
    unknown = [name for name in selected if name not in definitions]
    if unknown:
        parser.error(f"unknown case(s) for study {args.study}: {unknown}")
    for name in selected:
        if not args.force and (args.outdir / f"{name}.json").exists():
            print(f"skipping {name} (already measured; --force re-runs)", flush=True)
            continue
        print(f"running {name}", flush=True)
        row = run_case(
            name,
            checkpoint=args.outdir / f"{name}.checkpoint.npz",
            **definitions[name],
        )
        np.savez(args.outdir / f"{name}.npz", times=row["times"], probe=row["probe"])
        serial = _serial(row)
        (args.outdir / f"{name}.json").write_text(json.dumps(serial, indent=2) + "\n")
        print(json.dumps(serial, indent=2), flush=True)
        _write_study(args.outdir, args.study, definitions)
    _write_study(args.outdir, args.study, definitions)
    if all((args.outdir / f"{key}.json").exists() for key in definitions):
        _plot(args.outdir, args.study)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
