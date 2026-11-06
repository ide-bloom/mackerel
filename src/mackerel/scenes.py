"""Small command-line demo scenes.

The scenes deliberately use the same public ``Simulation`` API as the validation programs.
They write PNG frames/GIFs under ``outputs/`` and never require a notebook or a browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mackerel.advect import Scheme
from mackerel.diagnostics import vorticity_field
from mackerel.grid import (
    Array,
    BoundaryConditions,
    Grid,
    Kind,
    SideBC,
    apply_velocity_bc,
    walls_bc,
)
from mackerel.obstacles import apply_solid_velocity, circle_mask
from mackerel.render import RGBArray, save_frames_gif, save_image
from mackerel.simulate import Simulation


def _frame(field: Array, cmap: str, lo: float, hi: float, solid: np.ndarray | None = None) -> RGBArray:
    import matplotlib

    norm = np.clip((field - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    rgb = matplotlib.colormaps[cmap](norm)[:, :, :3]
    if solid is not None and solid.shape == field.shape:
        rgb[solid] = (0.05, 0.05, 0.05)
    return np.ascontiguousarray((np.flipud(rgb) * 255).astype(np.uint8))


def _vorticity_frame(sim: Simulation) -> RGBArray:
    w = vorticity_field(sim.u, sim.v, sim.grid, sim.bc)
    wc = 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])
    return _frame(wc, "RdBu_r", -8.0, 8.0, sim.solid)


def _speed_frame(sim: Simulation) -> RGBArray:
    uc = sim.grid.u_to_centers(sim.u)
    vc = sim.grid.v_to_centers(sim.v)
    return _frame(np.sqrt(uc * uc + vc * vc), "viridis", 0.0, 1.0, sim.solid)


def _scalar_frame(sim: Simulation) -> RGBArray:
    return _frame(sim.scalars[next(iter(sim.scalars))], "inferno", 0.0, 1.0, sim.solid)


def _finish(frames: list[RGBArray], outdir: Path, name: str, sim: Simulation) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    if not frames:
        frames.append(_speed_frame(sim))
    save_image(frames[-1], outdir / f"{name}.png")
    save_frames_gif(frames, outdir / f"{name}.gif", duration_ms=70)
    summary = {
        "scene": name,
        "time": sim.time,
        "steps": len(sim.history),
        "energy": sim.energy(),
        "final_divergence": sim.history[-1].div_max if sim.history else 0.0,
        "output": str(outdir),
    }
    (outdir / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def _cavity(n: int, steps: int, dt: float, outdir: Path) -> int:
    sim = Simulation.at_rest(
        Grid(n, n), walls_bc((1.0, 0.0)), nu=0.01, dt=dt,
        scheme=Scheme.MACCORMACK, pressure_tol=1e-8,
    )
    frames: list[RGBArray] = []
    for k in range(steps):
        sim.step()
        if k % max(1, steps // 40) == 0 or k == steps - 1:
            frames.append(_speed_frame(sim))
    return _finish(frames, outdir, "cavity", sim)


def _taylor_green(n: int, steps: int, dt: float, outdir: Path) -> int:
    sim = Simulation.taylor_green(
        Grid(n, n), nu=0.01, dt=dt, scheme=Scheme.MACCORMACK, pressure_tol=1e-8
    )
    frames: list[RGBArray] = []
    for k in range(steps):
        sim.step()
        if k % max(1, steps // 40) == 0 or k == steps - 1:
            frames.append(_vorticity_frame(sim))
    return _finish(frames, outdir, "taylor_green", sim)


def _cylinder(n: int, steps: int, dt: float, outdir: Path) -> int:
    grid = Grid(2 * n, n, lx=2.0)
    bc = BoundaryConditions(
        SideBC(Kind.INFLOW, velocity=(1.0, 0.0)), SideBC(Kind.OUTFLOW),
        SideBC(Kind.NO_SLIP), SideBC(Kind.NO_SLIP),
    )
    solid = circle_mask(grid, (0.35, 0.5), 0.08)
    sim = Simulation.at_rest(
        grid, bc, nu=0.0016, dt=dt, scheme=Scheme.MACCORMACK, pressure_tol=1e-8,
        pressure_maxiter=2000,
    )
    sim.set_solid(solid)
    sim.u[:] = 1.0
    apply_velocity_bc(sim.u, sim.v, bc)
    apply_solid_velocity(sim.u, sim.v, solid)
    frames: list[RGBArray] = []
    for k in range(steps):
        sim.step()
        if k % max(1, steps // 40) == 0 or k == steps - 1:
            frames.append(_vorticity_frame(sim))
    return _finish(frames, outdir, "cylinder", sim)


def _jet(n: int, steps: int, dt: float, outdir: Path) -> int:
    grid = Grid(2 * n, n, lx=2.0)
    bc = BoundaryConditions(
        SideBC(Kind.INFLOW, velocity=(1.0, 0.0), scalar=1.0), SideBC(Kind.OUTFLOW),
        SideBC(Kind.NO_SLIP), SideBC(Kind.NO_SLIP),
    )
    sim = Simulation.at_rest(
        grid, bc, nu=0.001, scalar_diffusivity=1e-4, dt=dt,
        scheme=Scheme.SEMI_LAGRANGIAN, pressure_tol=1e-8,
    )
    sim.u[:] = 1.0
    apply_velocity_bc(sim.u, sim.v, bc)
    sim.add_scalar("dye", 0.0)
    frames: list[RGBArray] = []
    for k in range(steps):
        sim.step()
        if k % max(1, steps // 40) == 0 or k == steps - 1:
            frames.append(_scalar_frame(sim))
    return _finish(frames, outdir, "jet", sim)


def run_scene(name: str, argv: list[str] | None = None) -> int:
    """Run a named scene with ``--n``, ``--steps``, ``--dt`` and ``--out`` options."""
    parser = argparse.ArgumentParser(prog=f"mackerel run {name}")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    outdir = args.out or Path("outputs") / name
    if name == "smoke":
        from validation.smoke import run_plume

        result = run_plume(args.n, end_time=args.steps * (args.dt or 0.005), dt=args.dt or 0.005)
        outdir.mkdir(parents=True, exist_ok=True)
        save_image(result["frames"][-1], outdir / "smoke.png")
        save_frames_gif(result["frames"], outdir / "smoke.gif", duration_ms=70)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, np.ndarray))}, indent=2))
        return 0
    if name == "moving":
        from validation.moving import run_moving

        result = run_moving(args.n, end_time=args.steps * (args.dt or 0.005), dt=args.dt or 0.005)
        outdir.mkdir(parents=True, exist_ok=True)
        save_image(result["frames"][-1], outdir / "moving.png")
        save_frames_gif(result["frames"], outdir / "moving.gif", duration_ms=80)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, np.ndarray))}, indent=2))
        return 0
    if name in {"dam_break", "splash", "sloshing"}:
        from validation.flip import TAU_SCALE, run_dam_break, run_demo

        water_dt = args.dt or 0.0005
        if name == "dam_break":
            result = run_dam_break(args.n, end_tau=args.steps * water_dt * TAU_SCALE,
                                   dt=water_dt, pic_blend=0.05)
        else:
            result = run_demo(name, n=args.n, end_time=args.steps * water_dt, dt=water_dt)
        outdir.mkdir(parents=True, exist_ok=True)
        save_image(result["frames"][-1], outdir / f"{name}.png")
        save_frames_gif(result["frames"], outdir / f"{name}.gif", duration_ms=70)
        particles = result["particles"]
        np.savez(outdir / f"{name}_particles.npz", x=particles.x, y=particles.y,
                 u=particles.u, v=particles.v)
        excluded = {"records", "frames", "particles", "water", "series"}
        summary = {k: v for k, v in result.items()
                   if k not in excluded and not isinstance(v, (list, np.ndarray))}
        summary["output"] = str(outdir)
        (outdir / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        print(json.dumps(summary, indent=2, default=float))
        return 0

    dt = args.dt or (0.0025 if name == "taylor_green" else 0.005)
    if name == "cavity":
        return _cavity(args.n, args.steps, dt, outdir)
    if name == "taylor_green":
        return _taylor_green(args.n, args.steps, dt, outdir)
    if name == "cylinder":
        return _cylinder(args.n, args.steps, dt, outdir)
    if name == "jet":
        return _jet(args.n, args.steps, dt, outdir)
    raise ValueError(f"unknown scene {name!r}")
