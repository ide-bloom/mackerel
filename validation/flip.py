"""Milestone 7: PIC/FLIP dam break, splash and sloshing validation.

The dam-break comparison uses the 14-point Martin--Moyce (1952) series in
``validation/data/martin_moyce_1952.csv``.  The initial column has width ``a`` and height
``2a`` in a ``4a x 3a`` closed domain.  We report the actual rightmost-particle front (plus
half a cell to approximate the containing-cell interface), without a gate-delay correction or
fitted parameter.  Both PIC and FLIP are run; their differences are retained in the report.

Run ``python -m validation.flip`` or ``mackerel validate flip``.  Outputs go to
``outputs/m7``.  The default is deliberately a modest 48x36 grid so the whole executable
validation, including two methods and two demonstrations, is practical on a CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig, dam_break_particles
from mackerel.grid import Grid
from mackerel.render import RGBArray, save_frames_gif, save_image

OUT = Path("outputs/m7")
DATA = Path(__file__).parent / "data" / "martin_moyce_1952.csv"
GRAVITY = 9.81
WIDTH = 0.25
TAU_SCALE = float(np.sqrt(2.0 * GRAVITY / WIDTH))


def load_reference(path: Path = DATA) -> tuple[np.ndarray, np.ndarray]:
    """Read the documented ``(tau, delta)`` experimental digitisation."""
    # ``names=True`` would mistake the first comment line (which contains commas) for a
    # header, so use the explicit seven-line provenance preamble instead.
    rows = np.loadtxt(path, delimiter=",", comments="#", skiprows=8, ndmin=2)
    return np.asarray(rows[:, 0], dtype=np.float64), np.asarray(rows[:, 1], dtype=np.float64)


def _particle_rgb(water: FreeSurfaceWater, scale: int = 4) -> RGBArray:
    """Rasterise particle occupancy into a y-up blue water image."""
    import matplotlib

    grid = water.grid
    p = water.particles
    occupancy, _, _ = np.histogram2d(
        p.y / grid.ly, p.x / grid.lx, bins=(grid.ny, grid.nx),
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    density = np.clip(occupancy / max(water.particles_per_cell, 1), 0.0, 1.0)
    # A dark background and a bright, opaque liquid make the free surface legible in a GIF.
    rgba = matplotlib.colormaps["Blues"](0.18 + 0.82 * density)[:, :, :3]
    rgba[density <= 0.0] = (0.025, 0.035, 0.065)
    rgb = np.ascontiguousarray((np.flipud(rgba) * 255).astype(np.uint8))
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def _record(water: FreeSurfaceWater) -> dict[str, float]:
    return {
        "time": float(water.time),
        "tau": float(water.time * TAU_SCALE),
        "front": water.front_position(),
        "delta": float((water.front_position() + 0.5 * water.grid.h) / WIDTH),
        "volume": water.water_volume(),
        "divergence": float(water.history[-1].divergence) if water.history else 0.0,
        "max_speed": float(max(np.max(np.abs(water.particles.u)), np.max(np.abs(water.particles.v)))),
        "liquid_cells": float(np.count_nonzero(water.fluid)),
    }


def run_dam_break(
    n: int = 48,
    end_tau: float = 2.96,
    dt: float = 0.0005,
    pic_blend: float = 0.05,
) -> dict[str, Any]:
    """Run one instantaneous-release dam break and return raw series and measurements."""
    if n % 4:
        raise ValueError("n must be divisible by four for the 4a x 3a domain")
    grid = Grid(n, 3 * n // 4, lx=1.0)
    particles = dam_break_particles(grid, width=WIDTH, height=2.0 * WIDTH)
    water = FreeSurfaceWater(
        grid, particles,
        config=WaterConfig(
            dt=dt, pic_blend=pic_blend, gravity=GRAVITY,
            pressure_tol=1e-8, pressure_maxiter=1200,
        ),
    )
    end_time = end_tau / TAU_SCALE
    records = [_record(water)]
    frames: list[RGBArray] = [_particle_rgb(water)]
    frame_times = [0.0]
    t0 = time.perf_counter()
    while water.time < end_time - 1e-14:
        water.step(min(water.dt_for_cfl(), end_time - water.time))
        records.append(_record(water))
        if water.time - frame_times[-1] >= 0.02 or water.time >= end_time - 1e-14:
            frames.append(_particle_rgb(water))
            frame_times.append(water.time)
    tau = np.array([r["tau"] for r in records])
    delta = np.array([r["delta"] for r in records])
    ref_tau, ref_delta = load_reference()
    valid = ref_tau <= tau[-1] + 1e-12
    predicted = np.interp(ref_tau[valid], tau, delta)
    errors = predicted - ref_delta[valid]
    return {
        "method": "PIC" if pic_blend == 1.0 else f"FLIP (PIC blend {pic_blend:g})",
        "pic_blend": pic_blend,
        "n": n,
        "grid": [grid.nx, grid.ny],
        "dt_max": dt,
        "end_tau": end_tau,
        "time": water.time,
        "steps": len(water.history),
        "wall_time": time.perf_counter() - t0,
        "initial_particles": len(particles),
        "final_particles": len(water.particles),
        "volume_initial": records[0]["volume"],
        "volume_final": records[-1]["volume"],
        "volume_relative_error": abs(records[-1]["volume"] - records[0]["volume"]) / records[0]["volume"],
        "max_divergence": max(r["divergence"] for r in records),
        "final_divergence": records[-1]["divergence"],
        "reference_count": int(np.count_nonzero(valid)),
        "front_rmse": float(np.sqrt(np.mean(errors**2))),
        "front_max_abs_error": float(np.max(np.abs(errors))),
        "front_mean_relative_error": float(np.mean(np.abs(errors[1:]) / ref_delta[valid][1:])) if len(errors) > 1 else 0.0,
        "reference_tau": ref_tau[valid],
        "reference_delta": ref_delta[valid],
        "predicted_reference_delta": predicted,
        "records": records,
        "frames": frames,
        "particles": water.particles.copy(),
        "water": water,
    }


def make_demo(kind: str, n: int = 40, dt: float = 0.0005) -> FreeSurfaceWater:
    """Create the splash or tilted-water sloshing initial condition."""
    if n % 4:
        raise ValueError("n must be divisible by four")
    grid = Grid(n, 3 * n // 4, lx=1.0)
    if kind == "splash":
        def region(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            pool = Y <= 0.18
            drop = (X - 0.52) ** 2 + (Y - 0.49) ** 2 <= 0.085**2
            return pool | drop
    elif kind == "sloshing":
        def region(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return Y <= 0.30 + 0.20 * (X - 0.5)
    else:
        raise ValueError(f"unknown water demo {kind!r}")
    particles = ParticleSet.from_region(grid, region, particles_per_cell=4, seed=7)
    water = FreeSurfaceWater(
        grid, particles,
        config=WaterConfig(dt=dt, pic_blend=0.05, gravity=GRAVITY,
                           pressure_tol=1e-8, pressure_maxiter=1200),
    )
    return water


def run_demo(kind: str, n: int = 40, end_time: float = 0.8, dt: float = 0.0005) -> dict[str, Any]:
    """Run a closed-box demonstration and retain occupancy frames and raw particles."""
    water = make_demo(kind, n, dt)
    records = [_record(water)]
    frames = [_particle_rgb(water)]
    frame_times = [0.0]
    t0 = time.perf_counter()
    while water.time < end_time - 1e-14:
        water.step(min(water.dt_for_cfl(), end_time - water.time))
        records.append(_record(water))
        if water.time - frame_times[-1] >= 0.025 or water.time >= end_time - 1e-14:
            frames.append(_particle_rgb(water))
            frame_times.append(water.time)
    arr = np.asarray([[r["time"], r["volume"], r["divergence"], r["max_speed"]] for r in records])
    return {
        "kind": kind,
        "n": n,
        "grid": [water.grid.nx, water.grid.ny],
        "time": water.time,
        "dt_max": dt,
        "steps": len(water.history),
        "wall_time": time.perf_counter() - t0,
        "initial_particles": len(water.particles),
        "final_particles": len(water.particles),
        "volume_initial": records[0]["volume"],
        "volume_final": records[-1]["volume"],
        "max_divergence": max(r["divergence"] for r in records),
        "final_divergence": records[-1]["divergence"],
        "max_speed": max(r["max_speed"] for r in records),
        "records": records,
        "series": arr,
        "frames": frames,
        "particles": water.particles.copy(),
        "water": water,
    }


def _write_series(path: Path, records: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _save_result_json(path: Path, result: dict[str, Any]) -> None:
    excluded = {"records", "frames", "particles", "water", "series"}
    scalar = {k: _scalar(v) for k, v in result.items() if k not in excluded and not isinstance(v, np.ndarray)}
    # Arrays of reference values are saved in NPZ/CSV, not duplicated as JSON arrays.
    (path).write_text(json.dumps(scalar, indent=2, default=_scalar) + "\n")


def _save_demo(path_stem: Path, result: dict[str, Any]) -> None:
    _save_result_json(path_stem.with_suffix(".json"), result)
    _write_series(path_stem.with_suffix(".csv"), result["records"])
    p = result["particles"]
    np.savez(path_stem.with_suffix(".npz"), x=p.x, y=p.y, u=p.u, v=p.v, series=result["series"])
    save_image(result["frames"][-1], path_stem.with_suffix(".png"))
    save_frames_gif(result["frames"], path_stem.with_suffix(".gif"), duration_ms=70)


def _save_dam_figures(outdir: Path, flip: dict[str, Any], pic: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_tau, ref_delta = load_reference()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    for result, label, color in ((flip, "95% FLIP", "tab:blue"), (pic, "PIC", "tab:orange")):
        rec = result["records"]
        ax.plot([r["tau"] for r in rec], [r["delta"] for r in rec], color=color, lw=1.5, label=label)
    ax.plot(ref_tau, ref_delta, "ko", ms=4, label="Martin-Moyce (1952)")
    ax.set(xlabel=r"$\tau=t\sqrt{2g/a}$", ylabel=r"$\delta=x_{front}/a$", title="Dam-break front")
    ax.grid(alpha=0.25)
    ax.legend()
    ax = axes[1]
    p = flip["particles"]
    ax.scatter(p.x, p.y, s=2, c="tab:blue", alpha=0.5, rasterized=True)
    ax.set(xlim=(0, 1), ylim=(0, 0.75), xlabel="x", ylabel="y", title="FLIP particles at final time", aspect="equal")
    fig.savefig(outdir / "dam_break.png", dpi=150)
    plt.close(fig)
    save_image(flip["frames"][-1], outdir / "dam_break_final.png")
    save_frames_gif(flip["frames"], outdir / "dam_break.gif", duration_ms=60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--end-tau", type=float, default=2.96)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--demo-time", type=float, default=0.8)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    flip = run_dam_break(args.n, args.end_tau, args.dt, pic_blend=0.05)
    pic = run_dam_break(args.n, args.end_tau, args.dt, pic_blend=1.0)
    _save_result_json(args.outdir / "dam_break_flip.json", flip)
    _save_result_json(args.outdir / "dam_break_pic.json", pic)
    _write_series(args.outdir / "dam_break_flip.csv", flip["records"])
    _write_series(args.outdir / "dam_break_pic.csv", pic["records"])
    for result, name in ((flip, "dam_break_flip"), (pic, "dam_break_pic")):
        p = result["particles"]
        np.savez(
            args.outdir / f"{name}_fields.npz", x=p.x, y=p.y, u=p.u, v=p.v,
            reference_tau=result["reference_tau"], reference_delta=result["reference_delta"],
            predicted_reference_delta=result["predicted_reference_delta"],
        )
    _save_dam_figures(args.outdir, flip, pic)
    splash = run_demo("splash", n=max(20, args.n - 8), end_time=args.demo_time, dt=args.dt)
    slosh = run_demo("sloshing", n=max(20, args.n - 8), end_time=args.demo_time, dt=args.dt)
    _save_demo(args.outdir / "splash", splash)
    _save_demo(args.outdir / "sloshing", slosh)
    summary = {
        "dam_break_flip": {k: v for k, v in flip.items() if k in {
            "front_rmse", "front_max_abs_error", "front_mean_relative_error", "max_divergence",
            "volume_relative_error", "wall_time", "steps",
        }},
        "dam_break_pic": {k: v for k, v in pic.items() if k in {
            "front_rmse", "front_max_abs_error", "front_mean_relative_error", "max_divergence",
            "volume_relative_error", "wall_time", "steps",
        }},
        "splash": {k: splash[k] for k in ("time", "steps", "max_divergence", "max_speed", "wall_time")},
        "sloshing": {k: slosh[k] for k in ("time", "steps", "max_divergence", "max_speed", "wall_time")},
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=_scalar) + "\n")
    print(json.dumps(summary, indent=2, default=_scalar))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
