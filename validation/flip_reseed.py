"""Conservative paired reseeding for the PIC/FLIP free-surface liquid.

Three questions are answered with measurements rather than assertions.

1. *Is it conservative?* Every reseeding exchange moves a particle; it never creates or
   destroys one.  Particle count and nominal volume are compared before and after each run.
2. *Does it remove a real sampling defect?* Interior holes (empty cells enclosed by liquid)
   are counted with the same rule in both arms of the comparison, so the effect is not an
   artefact of a changed metric.
3. *Does it change the answer?* The dam-break front is compared with the Martin--Moyce (1952)
   digitisation used since M7, and the falling-drop splash scene is compared with its own
   unreseeded control, so the benefit *and* the cost are both reported.  The tilted sloshing
   scene is retained as a documented null result: it develops essentially no interior defect,
   so reseeding is inert there.

The opt-in default in :class:`mackerel.flip.WaterConfig` means every earlier M7 result remains
exactly reproducible; this driver is the only place reseeding is switched on.

Run ``python -m validation.flip_reseed`` or ``mackerel validate flip_reseed``.  Outputs go to
``outputs/m7/reseed``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig
from mackerel.grid import Grid
from mackerel.render import save_frames_gif, save_image
from validation.flip import (
    GATE_OPENING_DELAY,
    _particle_rgb,
    _save_result_json,
    _write_series,
    density_snapshot,
    load_reference,
    make_demo,
    reseed_summary,
    run_dam_break,
)

OUT = Path("outputs/m7/reseed")
# The single documented reseeding configuration.  ``reseed_radius=5`` cells is a compromise:
# a tight radius leaves interior holes unfilled at the splash front, a loose one lets a
# particle jump several cells, which is the artefact this feature exists to avoid.
CONFIG: dict[str, Any] = {
    "reseed_interval": 1,
    "min_particles_per_cell": 2,
    "max_particles_per_cell": 8,
    "reseed_target": None,
    "reseed_interior_only": True,
    "reseed_radius": 5.0,
    "reseed_cooldown": 10,
    "reseed_seed": 0,
}

COLUMNS = ("time", "tau", "front", "delta", "volume", "divergence", "max_speed", "liquid_cells")


def _screen_exchange() -> dict[str, Any]:
    """A hand-built state that exercises every branch of the donor/recipient rule.

    A 12x9 grid holds a four-per-cell liquid slab covering cell columns 2..9 and rows 1..7.
    Three defects are planted in its interior: one cell is emptied completely (an interior
    hole), one is reduced to a single particle (a starved interior cell), and a third is given
    eight extra particles (a donor).  The free-surface fringe stays at four per cell.

    A correct exchange fills both defects, drains the donor back to the nominal density,
    ignores the fringe, keeps the particle count and volume, and moves nothing further than
    the configured radius.
    """
    grid = Grid(12, 9)
    gi, gj = np.meshgrid(np.arange(2, 10), np.arange(1, 8))
    ii, jj = gi.ravel(), gj.ravel()
    hole = (ii == 5) & (jj == 4)
    # The starved cell must not touch the hole, or it would legitimately become a fringe cell.
    starved = (ii == 6) & (jj == 6)
    x_parts, y_parts = [], []
    for k in range(4):
        ox = (k % 2 + 0.5) / 2
        oy = (k // 2 + 0.5) / 2
        keep = ~hole & ~(starved & (k > 0))
        x_parts.append((ii[keep] + ox) * grid.h)
        y_parts.append((jj[keep] + oy) * grid.h)
    extra_x = [(7 + 0.15 + 0.1 * k) * grid.h for k in range(8)]
    extra_y = [(4 + 0.2 + 0.1 * (k % 4)) * grid.h for k in range(8)]
    x = np.concatenate([*x_parts, extra_x])
    y = np.concatenate([*y_parts, extra_y])
    particles = ParticleSet(x, y, np.full_like(x, 0.25), np.full_like(y, -0.1))
    water = FreeSurfaceWater(
        grid, particles,
        config=WaterConfig(dt=0.001, reseed=True, **{**CONFIG, "reseed_cooldown": 0}),
    )
    planted = {"hole_cell": [5, 4], "starved_cell": [6, 6], "donor_cell": [7, 4]}
    before = density_snapshot(water)
    event = water.reseed()
    after = density_snapshot(water)
    return {
        "grid": [grid.nx, grid.ny],
        "planted": planted,
        "particles_before": float(len(particles)),
        "particles_after": float(len(water.particles)),
        "volume_before": before["volume"],
        "volume_after": after["volume"],
        "moved": event["moved"],
        "max_move_distance_over_h": event["max_move_distance"] / grid.h,
        "before": before,
        "after": after,
        "event": {key: value for key, value in event.items()
                  if not isinstance(value, dict) and isinstance(value, float)},
    }


def _run_demo(kind: str, n: int, end_time: float, reseed: bool) -> dict[str, Any]:
    """One closed-box demo run, with or without reseeding, keeping the frames for a GIF."""
    water = make_demo(kind, n=n, dt=0.0005, reseed=reseed, reseed_config=CONFIG)
    initial = density_snapshot(water)
    frames = [_particle_rgb(water)]
    frame_times = [0.0]
    # The interface rasters are saved as versioned PNGs so the paired phase comparison in
    # ``_phase_error`` reads exactly the bytes this run produced, not a re-encoded GIF.
    rasters = [frames[0].copy()]
    times = [0.0]
    records: list[dict[str, float]] = []
    import time as _time

    start = _time.perf_counter()
    while water.time < end_time - 1e-14:
        water.step(min(water.dt_for_cfl(), end_time - water.time))
        records.append({
            "time": float(water.time),
            "max_speed": float(water.history[-1].max_speed),
            "divergence": float(water.history[-1].divergence),
            "liquid_cells": float(water.history[-1].liquid_cells),
        })
        if water.time - frame_times[-1] >= 0.02 or water.time >= end_time - 1e-14:
            frames.append(_particle_rgb(water))
            rasters.append(frames[-1].copy())
            times.append(water.time)
            frame_times.append(water.time)
    final = density_snapshot(water)
    return {
        "kind": kind,
        "n": n,
        "reseed": reseed,
        "reseed_config": CONFIG if reseed else None,
        "steps": float(len(water.history)),
        "wall_time": _time.perf_counter() - start,
        "initial": initial,
        "final": final,
        "peak_speed": float(max(r["max_speed"] for r in records)),
        "rms_speed": float(np.sqrt(np.mean([r["max_speed"] ** 2 for r in records]))),
        "max_divergence": float(max(r["divergence"] for r in records)),
        "particle_speed_rms": float(np.sqrt(np.mean(water.particles.u**2 + water.particles.v**2))),
        "records": records,
        "frames": frames,
        "rasters": rasters,
        "raster_times": times,
        "reseed_summary": reseed_summary(water),
        "particles": water.particles.copy(),
    }


def _read_series(path: Path) -> np.ndarray:
    """Read a saved dam-break CSV into an array ordered like :data:`COLUMNS`."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    unexpected = [key for key in rows[0] if key not in COLUMNS]
    if unexpected:
        raise ValueError(f"unexpected columns in {path}: {unexpected}")
    return np.asarray([[float(row[key]) for key in COLUMNS] for row in rows], dtype=np.float64)


def _grid_label(n: int) -> str:
    """``48x36`` style label for an ``n x 3n/4`` dam-break grid."""
    return f"{n}x{3 * n // 4}"


def _scalar_result(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {"records", "frames", "particles", "water", "series", "reference_tau",
                "reference_delta", "predicted_reference_delta"}
    return {key: value for key, value in result.items()
            if key not in excluded and not isinstance(value, np.ndarray)}


def _save_dam_break(outdir: Path, label: str, result: dict[str, Any]) -> dict[str, Any]:
    row = _scalar_result(result)
    row["case"] = label
    _save_result_json(outdir / f"dam_break_{label}.json", result)
    (outdir / f"dam_break_{label}.json").write_text(json.dumps(row, indent=2) + "\n")
    _write_series(outdir / f"dam_break_{label}.csv", result["records"])
    particles = result["particles"]
    np.savez(
        outdir / f"dam_break_{label}.npz",
        x=particles.x, y=particles.y, u=particles.u, v=particles.v,
        reference_tau=result["reference_tau"], reference_delta=result["reference_delta"],
        predicted_reference_delta=result["predicted_reference_delta"],
    )
    save_image(result["frames"][-1], outdir / f"dam_break_{label}.png")
    save_frames_gif(result["frames"], outdir / f"dam_break_{label}.gif", duration_ms=60)
    return row


def _save_demo(outdir: Path, label: str, run: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in run.items()
           if key not in {"records", "frames", "particles", "rasters", "raster_times"}}
    kind = run["kind"]
    np.savez(
        outdir / f"{kind}_{label}_frames.npz",
        frames=np.asarray(run["rasters"], dtype=np.uint8),
        times=np.asarray(run["raster_times"], dtype=np.float64),
    )
    particles = run["particles"]
    np.savez(
        outdir / f"{kind}_{label}.npz",
        x=particles.x, y=particles.y, u=particles.u, v=particles.v,
        series=np.asarray([[r["time"], r["max_speed"], r["divergence"], r["liquid_cells"]]
                           for r in run["records"]]),
    )
    (outdir / f"{kind}_{label}.json").write_text(json.dumps(row, indent=2) + "\n")
    with (outdir / f"{kind}_{label}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "max_speed", "divergence",
                                                    "liquid_cells"])
        writer.writeheader()
        writer.writerows(run["records"])
    save_image(run["frames"][-1], outdir / f"{kind}_{label}.png")
    save_frames_gif(run["frames"], outdir / f"{kind}_{label}.gif", duration_ms=70)
    return row


def _saved_frames(outdir: Path, kind: str, label: str) -> list[np.ndarray]:
    """Read back the versioned interface rasters of one demo arm."""
    with np.load(outdir / f"{kind}_{label}_frames.npz") as data:
        return [np.asarray(frame) for frame in data["frames"]]


def _phase_error(frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> dict[str, float]:
    """Compare the interface motion of a paired demo run.

    Both arms share the same particles, seed and time grid, so they can only differ through
    reseeding.  A pixel is liquid when its raster value exceeds a fixed threshold; the fraction
    of pixels where the two classified masks disagree measures how far the interface has drifted.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError("paired runs must save the same frames")
    differences = []
    for a, b in zip(frames_a, frames_b, strict=True):
        liquid_a = np.any(a > 90, axis=2)
        liquid_b = np.any(b > 90, axis=2)
        differences.append(float(np.mean(liquid_a != liquid_b)))
    differing = [value for value in differences if value > 0.0]
    return {
        "frames": float(len(differences)),
        "identical_frames": float(sum(value == 0.0 for value in differences)),
        "first_frame_mismatch": differences[0],
        "last_frame_mismatch": differences[-1],
        "max_frame_mismatch": float(max(differences)),
        "mean_frame_mismatch": float(np.mean(differences)),
        "mean_mismatch_where_different": float(np.mean(differing)) if differing else 0.0,
    }


def _plot(outdir: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_tau, ref_delta = load_reference()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    cases = report["dam_break_cases"]
    coarse = f"control_{report['grids'][0]}"
    fine = f"control_{report['grids'][-1]}"
    ax = axes[0]
    for label, color, style in ((coarse, "tab:blue", "-"), (fine, "tab:cyan", "-")):
        series = _read_series(outdir / f"dam_break_{label}.csv")
        ax.plot(series[:, 1], series[:, 3], color=color, ls=style, lw=1.4,
                label=label.replace("_", " ") + " (no reseeding)")
    for label, color in ((coarse.replace("control", "reseed"), "tab:green"),
                         (fine.replace("control", "reseed"), "tab:orange")):
        series = _read_series(outdir / f"dam_break_{label}.csv")
        ax.plot(series[:, 1], series[:, 3], color=color, ls="--", lw=1.2,
                label=label.replace("_", " ") + " (reseeded)")
    ax.plot(ref_tau, ref_delta, "ko", ms=4, label="Martin-Moyce (1952)")
    ax.set(xlabel="tau", ylabel="delta", title="Dam-break front")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = sorted(report["dam_break_cases"],
                    key=lambda key: (int(key.rsplit("_", 1)[-1]), key))
    values = [report["dam_break_cases"][label]["front_rmse"] for label in labels]
    ax.bar(labels, values, color=["tab:blue" if "control" in label else "tab:green"
                                  for label in labels], edgecolor="black", linewidth=0.4)
    ax.set(ylabel="front RMSE", title="Front error versus grid")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    keys = ("interior_holes", "interior_starved", "fringe_starved")
    bars = [(f"control_{n}", "tab:blue", -0.19) for n in report["grids"]]
    bars += [(f"reseed_{n}", "tab:green", 0.01) for n in report["grids"]]
    bars = [(label, color, offset) for label, color, offset in bars
            if label in cases]
    for label, color, offset in bars:
        deficits = cases[label]["final_deficits"]
        ax.bar(np.arange(len(keys)) + offset, [deficits[key] for key in keys], width=0.19,
               color=color, edgecolor="black", linewidth=0.3,
               label=label.replace("_", " "))
    ax.set_xticks(np.arange(len(keys)), [key.replace("_", "\n") for key in keys])
    ax.set(ylabel="cells", title=f"Final sampling defects, {report['labels'][-1]}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "reseed_summary.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--n", type=int, default=96, help="fine dam-break grid width")
    parser.add_argument("--coarse-n", type=int, default=48, help="coarse dam-break grid width")
    parser.add_argument("--end-tau", type=float, default=2.96)
    parser.add_argument("--demo-time", type=float, default=0.4)
    parser.add_argument("--demo-n", type=int, default=20)
    parser.add_argument("--skip-demos", action="store_true")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": CONFIG,
        "gate_opening_delay": GATE_OPENING_DELAY,
        "protocol": ("a=.25, initial height 2a, g=9.81, dt_max=.0005, 95% FLIP, "
                     "front = rightmost particle + h/2"),
        "grids": [args.coarse_n, args.n],
        "labels": [_grid_label(args.coarse_n), _grid_label(args.n)],
        "conservation_rule": "reseeding moves particles only; count and nominal volume are exact",
        "screen": _screen_exchange(),
    }
    print("screen", json.dumps(report["screen"]["event"], indent=2), flush=True)

    cases: dict[str, Any] = {}
    for n in (args.coarse_n, args.n):
        for reseed in (False, True):
            label = f"{'reseed' if reseed else 'control'}_{n}"
            print(f"running dam break {label}", flush=True)
            result = run_dam_break(
                n=n, end_tau=args.end_tau, pic_blend=0.05,
                reseed=reseed, reseed_config=CONFIG,
            )
            cases[label] = _save_dam_break(args.outdir, label, result)
            print(json.dumps({key: cases[label][key] for key in (
                "front_rmse", "front_max_abs_error", "front_mean_relative_error",
                "volume_relative_error", "steps", "final_particles", "wall_time",
            )}, indent=2), flush=True)
    report["dam_break_cases"] = cases

    demos: dict[str, Any] = {}
    if not args.skip_demos:
        for kind in ("splash", "sloshing"):
            arms: dict[str, Any] = {}
            for reseed in (False, True):
                label = "reseed" if reseed else "control"
                print(f"running {kind} {label}", flush=True)
                run = _run_demo(kind, n=args.demo_n, end_time=args.demo_time, reseed=reseed)
                arms[label] = _save_demo(args.outdir, label, run)
                print(json.dumps({key: arms[label][key] for key in (
                    "steps", "peak_speed", "rms_speed", "max_divergence",
                )}, indent=2), flush=True)
            arms["pairing"] = _phase_error(
                _saved_frames(args.outdir, kind, "control"),
                _saved_frames(args.outdir, kind, "reseed"),
            )
            demos[kind] = arms
    report["demos"] = demos

    (args.outdir / "reseed_summary.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    _plot(args.outdir, report)
    print(json.dumps({label: case["front_rmse"] for label, case in cases.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
