"""Why the dam-break front error is not a grid-convergence trend.

The milestone study measured FLIP front RMSE rising from 0.0726 (48x36) to 0.2565 (96x72)
while PIC fell from 0.6225 to 0.0868.  This driver isolates which of the plausible causes
actually explains that, one switch at a time, and adds a third resolution (144x108).

Everything is one-at-a-time against a baseline of ``n=96`` (95% FLIP, ``dt_max=5e-4``, four
particles per cell, seed 0, rightmost-particle-plus-half-cell front, zero gate delay).  No
parameter is tuned: each alternative is a documented setting that the milestone reports
already used or that the front definition itself implies.

Groups
------
1. ``front_definition`` -- the same P2 trajectory scored with seven front conventions, all
   non-dimensionalised by ``a``.  Is the ranking an artefact of the convention?
2. ``controls`` -- jitter seed, particles per cell, ``dt_max`` and PIC blend, each varied
   alone, scored with the same conventions.
3. ``resolution`` -- 48x36, 96x72 and 144x108, plus a ``dt`` refinement at fixed ``n`` so
   temporal and spatial error can be separated.

Run ``python -m validation.dam_break_diagnosis``.  Outputs go to ``outputs/m7/diagnosis``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig, dam_break_particles
from mackerel.grid import Grid
from validation.flip import TAU_SCALE, WIDTH, load_reference

OUT = Path("outputs/m7/diagnosis")
GRAVITY = 9.81
END_TAU = 2.96
# Every convention is reported; the milestone headline is ``rightmost_half``.
FRONT_KEYS = (
    "rightmost", "rightmost_half", "p99", "p995", "column_k4", "column_k8", "thickness_2h",
)


def front_metrics(water: FreeSurfaceWater) -> dict[str, float]:
    """Seven front estimators for the same particle state, each in physical ``x``.

    ``rightmost`` is the raw maximum; ``rightmost_half`` adds the half-cell containment offset
    used since M7.  The percentile estimators are robust to a single fast particle.
    ``column_k4``/``column_k8`` take the last cell column holding at least four or eight
    particles, and ``thickness_2h`` the last column whose liquid layer is at least two cells
    deep, which is the closest analogue of a laboratory interface probe.
    """
    grid = water.grid
    h = grid.h
    x = water.particles.x
    counts = water._particle_counts()
    occupied = counts > 0
    column_particles = counts.sum(axis=0)
    column_height = occupied.sum(axis=0) * h

    def last_column(mask: np.ndarray) -> float:
        index = np.nonzero(mask)[0]
        return float((index[-1] + 1) * h) if index.size else 0.0

    return {
        "rightmost": float(x.max()),
        "rightmost_half": float(x.max() + 0.5 * h),
        "p99": float(np.percentile(x, 99.0)),
        "p995": float(np.percentile(x, 99.5)),
        "column_k4": last_column(column_particles >= 4),
        "column_k8": last_column(column_particles >= 8),
        "thickness_2h": last_column(column_height >= 2.0 * h),
    }


def run_case(
    n: int,
    dt: float = 5e-4,
    pic_blend: float = 0.05,
    particles_per_cell: int = 4,
    seed: int = 0,
    end_tau: float = END_TAU,
) -> dict[str, Any]:
    """Run one dam break and record every front estimator at each step."""
    grid = Grid(n, 3 * n // 4, lx=1.0)
    if particles_per_cell == 4 and seed == 0:
        particles = dam_break_particles(grid, width=WIDTH, height=0.5)
    else:
        particles = ParticleSet.from_region(
            grid, lambda X, Y: (X <= WIDTH) & (Y <= 0.5),
            particles_per_cell=particles_per_cell, seed=seed,
        )
    water = FreeSurfaceWater(
        grid, particles, particles_per_cell=particles_per_cell,
        config=WaterConfig(dt=dt, pic_blend=pic_blend, gravity=GRAVITY,
                           pressure_tol=1e-8, pressure_maxiter=1200),
    )
    tau: list[float] = []
    values: dict[str, list[float]] = {key: [] for key in FRONT_KEYS}
    wall_hit_tau: float | None = None
    start = time.perf_counter()
    end_time = end_tau / TAU_SCALE
    while water.time < end_time - 1e-14:
        water.step(min(water.dt_for_cfl(), end_time - water.time))
        metrics = front_metrics(water)
        tau.append(float(water.time * TAU_SCALE))
        for key in FRONT_KEYS:
            values[key].append(metrics[key])
        if wall_hit_tau is None and water.front_position() > 1.0 - 3.0 * grid.h:
            wall_hit_tau = float(water.time * TAU_SCALE)
    return {
        "n": n,
        "grid": [grid.nx, grid.ny],
        "dt_max": dt,
        "cells_per_column": float(n),
        "pic_blend": pic_blend,
        "particles_per_cell": particles_per_cell,
        "seed": seed,
        "steps": len(water.history),
        "wall_time": time.perf_counter() - start,
        "particles": float(len(water.particles)),
        "volume_relative_error": abs(water.water_volume() - 0.5 * WIDTH) / (0.5 * WIDTH),
        "max_divergence": float(max(info.divergence for info in water.history)),
        "wall_hit_tau": wall_hit_tau,
        "tau": tau,
        "fronts": values,
    }


def score(result: dict[str, Any]) -> dict[str, float]:
    """RMSE of every front convention against the 14 Martin--Moyce stations.

    Both the measured front and the reference are divided by ``a``, so the half-cell offset
    convention is applied to the measurement only, exactly as the milestone reports do.
    """
    reference_tau, reference_delta = load_reference()
    tau = np.asarray(result["tau"])
    valid = reference_tau <= tau[-1] + 1e-12
    out: dict[str, float] = {}
    for key, series in result["fronts"].items():
        measured = np.asarray(series) / WIDTH
        predicted = np.interp(reference_tau[valid], tau, measured)
        out[key] = float(np.sqrt(np.mean((predicted - reference_delta[valid]) ** 2)))
    return out


def _serial(result: dict[str, Any], label: str, group: str) -> dict[str, Any]:
    row = {key: value for key, value in result.items() if key not in {"tau", "fronts"}}
    row["case"] = label
    row["group"] = group
    row["rmse"] = score(result)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--fast", action="store_true", help="skip the 144x108 and fine-dt cases")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    def record(label: str, group: str, **kwargs: Any) -> dict[str, Any]:
        print(f"running {label}", flush=True)
        result = run_case(**kwargs)
        row = _serial(result, label, group)
        rows.append(row)
        print(json.dumps({"case": label, "rmse": {k: round(v, 5) for k, v in row["rmse"].items()},
                          "steps": row["steps"], "wall_hit_tau": row["wall_hit_tau"],
                          "wall": round(row["wall_time"], 1)}, indent=2), flush=True)
        arrays: dict[str, Any] = {"tau": np.asarray(result["tau"], dtype=np.float64)}
        for key, values in result["fronts"].items():
            arrays[f"front_{key}"] = np.asarray(values, dtype=np.float64)
        np.savez(args.outdir / f"case_{label}.npz", **arrays)
        return row

    # 1. The three grids in the milestone study, plus the third resolution.
    record("baseline_48", "resolution", n=48)
    record("baseline_96", "resolution", n=96)
    if not args.fast:
        record("baseline_144", "resolution", n=144)

    # 2. One-at-a-time controls on the 96 grid, which is where the regression appears.
    record("seed_1", "controls", n=96, seed=1)
    record("seed_2", "controls", n=96, seed=2)
    record("ppc_9", "controls", n=96, particles_per_cell=9)
    record("dt_half", "controls", n=96, dt=2.5e-4)
    if not args.fast:
        record("dt_quarter", "controls", n=96, dt=1.25e-4)
    record("blend_0", "controls", n=96, pic_blend=0.0)
    record("blend_20", "controls", n=96, pic_blend=0.2)
    record("pic_96", "controls", n=96, pic_blend=1.0)
    record("pic_48", "controls", n=48, pic_blend=1.0)

    # 3. Temporal convergence at fixed grid, to separate time error from space error.
    if not args.fast:
        record("dt_half_48", "resolution", n=48, dt=2.5e-4)
        record("dt_half_144", "resolution", n=144, dt=2.5e-4)

    report = {
        "protocol": ("a=.25, height 2a, g=9.81, instantaneous release, tau=(t+0)*sqrt(2g/a), "
                     "14 Martin-Moyce stations, front normalised by a"),
        "front_conventions": list(FRONT_KEYS),
        "headline_convention": "rightmost_half",
        "cases": rows,
    }
    (args.outdir / "dam_break_diagnosis.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    _plot(args.outdir, rows)
    return 0


def _plot(outdir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    # (a) convention sweep: is the ranking robust to the front definition?
    ax = axes[0]
    width = 0.26
    positions = np.arange(len(FRONT_KEYS))
    groups = [("baseline_48", "48x36", "tab:blue"), ("baseline_96", "96x72", "tab:orange"),
              ("baseline_144", "144x108", "tab:green")]
    for offset, (label, name, color) in zip((-width, 0.0, width), groups, strict=False):
        row = next((r for r in rows if r["case"] == label), None)
        if row is None:
            continue
        ax.bar(positions + offset, [row["rmse"][key] for key in FRONT_KEYS], width,
               color=color, label=name)
    ax.set_xticks(positions, [key.replace("_", "\n") for key in FRONT_KEYS], fontsize=8)
    ax.set(ylabel="front RMSE", title="Front definition versus grid")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # (b) controls on the 96 grid.
    ax = axes[1]
    controls = [r for r in rows if r["group"] == "controls" and r["n"] == 96]
    names = [r["case"] for r in controls]
    ax.barh(names, [r["rmse"]["rightmost_half"] for r in controls], color="tab:purple")
    base = next((r for r in rows if r["case"] == "baseline_96"), None)
    if base is not None:
        ax.axvline(base["rmse"]["rightmost_half"], color="black", ls="--", lw=1,
                   label="baseline 96x72")
        ax.legend(fontsize=8)
    ax.set(xlabel="front RMSE (rightmost + h/2)", title="One-at-a-time controls, 96x72")
    ax.grid(axis="x", alpha=0.25)

    # (c) trajectories: where the error actually comes from.
    ax = axes[2]
    reference_tau, reference_delta = load_reference()
    ax.plot(reference_tau, reference_delta, "ko", ms=4, label="Martin-Moyce (1952)")
    for label, name, color in groups:
        path = outdir / f"case_{label}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            tau = data["tau"]
            front = data["front_rightmost_half"] / WIDTH
        ax.plot(tau, front, color=color, lw=1.4, label=name)
    ax.set(xlabel="tau", ylabel="delta = front / a", title="Overshoot and phase")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "dam_break_diagnosis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
