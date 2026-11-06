"""Curated showcase media for ``README.md``.

This is the only script that writes into ``docs/media/``, which ``.gitignore`` whitelists for
tracked GIF/PNG files.  Every image comes from the same validated solvers the milestone
reports use; nothing is drawn by hand.  Resolution and duration are deliberately modest,
because this media is a preview and not a substitute for the validation artifacts in
``outputs/``.

Run ``python -m validation.showcase`` (or ``PYTHONPATH=. mackerel validate showcase``), or
``--only splash`` for one image.  Every scene is seeded and configured here, so a re-run is
deterministic, and ``docs/media/showcase.json`` records the parameters and measured diagnostics
that produced each file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme
from mackerel.diagnostics import vorticity_field
from mackerel.flip import FreeSurfaceWater, WaterConfig, dam_break_particles
from mackerel.grid import Grid
from mackerel.render import RGBArray, save_frames_gif, save_image
from mackerel.simulate import Simulation

MEDIA = Path("docs/media")

# Palette constants shared by every water frame, so the liquid reads the same way everywhere:
# a bright rim where a liquid cell touches air, a lighter body for sparse cells, and a dark,
# clearly empty box behind it.
AIR = (0.035, 0.045, 0.075)
WATER_BODY = np.array([0.13, 0.40, 0.72])
WATER_SPARSE = np.array([0.05, 0.18, 0.38])
WATER_SURFACE = (0.55, 0.86, 1.0)

# Fixed preview configuration.  ``dt`` is the physical time step; it is deliberately *not*
# derived from ``end_time / steps`` so the frames sample the motion evenly.
PREVIEW: dict[str, dict[str, float]] = {
    "dam_break": {"n": 48, "dt": 0.00025, "end_time": 0.34, "frame_interval": 0.0125,
                  "duration_ms": 60, "scale": 6},
    "splash": {"n": 64, "dt": 0.0005, "end_time": 0.30, "frame_interval": 0.012,
               "duration_ms": 55, "scale": 6},
    "sloshing": {"n": 64, "dt": 0.0005, "end_time": 0.42, "frame_interval": 0.015,
                 "duration_ms": 55, "scale": 6},
    "smoke": {"n": 64, "dt": 0.005, "end_time": 2.4, "duration_ms": 60, "scale": 5},
    # The vorticity field is smooth, so it compresses badly as a GIF: fewer frames at a smaller
    # pixel scale keep the committed file small without hiding the decay.
    "taylor_green": {"n": 64, "dt": 0.0025, "end_time": 1.2, "frame_interval": 0.10,
                     "duration_ms": 70, "scale": 4},
}


def water_rgb(water: FreeSurfaceWater, scale: int) -> RGBArray:
    """Render particle occupancy: bulk liquid, sparse interior, and a bright free-surface rim."""
    counts = water._particle_counts()
    occupied = counts > 0
    density = np.clip(counts / max(float(water.particles_per_cell), 1.0), 0.0, 1.0)
    img = np.full((*counts.shape, 3), AIR, dtype=np.float64)
    img = np.where(
        occupied[..., None], WATER_SPARSE + (WATER_BODY - WATER_SPARSE) * density[..., None], img
    )
    neighbours = np.zeros_like(counts)
    neighbours[1:, :] += occupied[:-1, :]
    neighbours[:-1, :] += occupied[1:, :]
    neighbours[:, 1:] += occupied[:, :-1]
    neighbours[:, :-1] += occupied[:, 1:]
    # A liquid cell with fewer than four liquid neighbours touches the exterior air region.
    img[occupied & (neighbours < 4)] = WATER_SURFACE
    rgb = np.ascontiguousarray((np.flipud(img) * 255).astype(np.uint8))
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def field_rgb(
    values: np.ndarray, cmap: str, lo: float, hi: float,
    solid: np.ndarray | None = None, scale: int = 5,
) -> RGBArray:
    """Colour-mapped scalar raster with an optional dark solid mask, y pointing up."""
    import matplotlib

    norm = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    rgb = matplotlib.colormaps[cmap](norm)[:, :, :3].astype(np.float64)
    if solid is not None and solid.shape == values.shape:
        rgb[solid] = (0.1, 0.1, 0.1)
    out = np.ascontiguousarray((np.flipud(rgb) * 255).astype(np.uint8))
    return np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)


def _vorticity(sim: Simulation, scale: int, lo: float = -4.0, hi: float = 4.0) -> RGBArray:
    """Diverging vorticity, clipped to a fixed range so the frames stay comparable."""
    w = vorticity_field(sim.u, sim.v, sim.grid, sim.bc)
    cell = 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])
    return field_rgb(cell, "RdBu_r", lo, hi, sim.solid, scale)


def _write(frames: list[RGBArray], path: Path, duration_ms: float) -> Path:
    gif = save_frames_gif(frames, path, duration_ms=int(duration_ms))
    save_image(frames[-1], path.with_suffix(".png"))
    return gif


def render_dam_break(outdir: Path) -> dict[str, Any]:
    """95% FLIP dam break: the column collapses and runs out along the floor."""
    spec = PREVIEW["dam_break"]
    grid = Grid(int(spec["n"]), 3 * int(spec["n"]) // 4)
    water = FreeSurfaceWater(
        grid, dam_break_particles(grid, width=0.25, height=0.5),
        config=WaterConfig(dt=spec["dt"], pic_blend=0.05, pressure_tol=1e-8,
                           pressure_maxiter=1200),
    )
    frames = [water_rgb(water, int(spec["scale"]))]
    next_time = 0.0
    while water.time < spec["end_time"] - 1e-14:
        water.step(min(water.dt_for_cfl(), spec["end_time"] - water.time))
        if water.time >= next_time:
            frames.append(water_rgb(water, int(spec["scale"])))
            next_time += spec["frame_interval"]
    gif = _write(frames, outdir / "dam_break.gif", spec["duration_ms"])
    return {
        "file": str(gif), "frames": len(frames), "grid": [grid.nx, grid.ny],
        "steps": len(water.history), "dt": spec["dt"], "end_time": water.time,
        "front": water.front_position(),
        "max_divergence": max(info.divergence for info in water.history),
        "volume_relative_error": abs(water.water_volume() - 0.25 * 0.5) / (0.25 * 0.5),
        "note": "front = rightmost particle; see docs/milestones/M7.md for the reference study",
    }


def _render_demo(kind: str, outdir: Path) -> dict[str, Any]:
    """Splash or sloshing: the seeded closed-box demos from ``validation.flip.make_demo``."""
    from validation.flip import make_demo

    spec = PREVIEW[kind]
    water = make_demo(kind, n=int(spec["n"]), dt=spec["dt"])
    frames = [water_rgb(water, int(spec["scale"]))]
    next_time = 0.0
    while water.time < spec["end_time"] - 1e-14:
        water.step(min(water.dt_for_cfl(), spec["end_time"] - water.time))
        if water.time >= next_time:
            frames.append(water_rgb(water, int(spec["scale"])))
            next_time += spec["frame_interval"]
    gif = _write(frames, outdir / f"{kind}.gif", spec["duration_ms"])
    return {
        "file": str(gif), "frames": len(frames),
        "grid": [water.grid.nx, water.grid.ny], "steps": len(water.history),
        "dt": spec["dt"], "end_time": water.time,
        "max_divergence": max(info.divergence for info in water.history),
        "particles": len(water.particles),
    }


def render_splash(outdir: Path) -> dict[str, Any]:
    """A drop falling into a pool: the clearest demonstration of the particle liquid."""
    return _render_demo("splash", outdir)


def render_sloshing(outdir: Path) -> dict[str, Any]:
    """Tilted water settling: a calmer free-surface case than the splash."""
    return _render_demo("sloshing", outdir)


def render_smoke(outdir: Path) -> dict[str, Any]:
    """Buoyant dye plume: scalar advection and buoyancy next to the velocity cases."""
    from validation.smoke import run_plume

    spec = PREVIEW["smoke"]
    result = run_plume(int(spec["n"]), end_time=spec["end_time"], dt=spec["dt"])
    frames = result["frames"]
    if len(frames) < 2:
        raise RuntimeError("the plume recorded a single frame; check validation/smoke.py")
    # The plume frames are native grid resolution; upscale so the GIF is legible at README size.
    scale = int(spec["scale"])
    frames = [np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1) for frame in frames]
    gif = _write(frames, outdir / "smoke.gif", spec["duration_ms"])
    return {
        "file": str(gif), "frames": len(frames), "grid": [int(spec["n"]), int(spec["n"])],
        "steps": result["steps"], "end_time": result["time"],
        "max_divergence": result["max_divergence"],
        "centre_rise": result["final_centre_y"] - result["initial_centre_y"],
    }


def render_taylor_green(outdir: Path) -> dict[str, Any]:
    """Decaying Taylor-Green vortices: the canonical advection/diffusion picture."""
    spec = PREVIEW["taylor_green"]
    sim = Simulation.taylor_green(
        Grid(int(spec["n"]), int(spec["n"])), nu=0.01, dt=spec["dt"],
        scheme=Scheme.MACCORMACK, pressure_tol=1e-8,
    )
    frames = [_vorticity(sim, int(spec["scale"]))]
    next_time = 0.0
    while sim.time < spec["end_time"] - 1e-12:
        sim.step()
        if sim.time >= next_time:
            frames.append(_vorticity(sim, int(spec["scale"])))
            next_time += spec["frame_interval"]
    gif = _write(frames, outdir / "taylor_green.gif", spec["duration_ms"])
    return {
        "file": str(gif), "frames": len(frames), "grid": [sim.grid.nx, sim.grid.ny],
        "steps": len(sim.history), "end_time": sim.time, "nu": 0.01,
        "final_divergence": sim.history[-1].div_max,
    }


def build_hero(outdir: Path) -> dict[str, Any]:
    """One wide still for the top of the README: liquid on the left, vorticity on the right.

    Both panels are square rasters of the same size so neither is stretched, and the whole
    scene is kept at ``scale=5`` so the particle liquid is legible at README width.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from validation.flip import make_demo

    water = make_demo("splash", n=96, dt=0.0005)
    # Land on the crown: the drop has hit, the walls are rising and the central cavity is
    # still open.  Earlier frames just show a falling circle; later ones are already calm.
    while water.time < 0.33 - 1e-14:
        water.step(min(water.dt_for_cfl(), 0.33 - water.time))
    liquid = water_rgb(water, scale=5)

    sim = Simulation.taylor_green(
        Grid(64, 64), nu=0.01, dt=0.0025, scheme=Scheme.MACCORMACK, pressure_tol=1e-8
    )
    sim.run_steps(60)
    vortices = _vorticity(sim, scale=5)

    # The liquid domain is 4:3 and the vorticity box is square; matching the column widths to
    # those ratios keeps both panels the same height and neither image stretched.
    fig, axes = plt.subplots(
        1, 2, figsize=(9.6, 4.4), layout="constrained",
        gridspec_kw={"width_ratios": [4, 3]},
    )
    axes[0].imshow(liquid, interpolation="nearest")
    axes[0].set(title="PIC/FLIP liquid: splash crown at t=0.33")
    axes[1].imshow(vortices, interpolation="nearest")
    axes[1].set(title="Taylor-Green vorticity at t=0.15")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(outdir / "hero.png", dpi=110)
    plt.close(fig)
    return {
        "file": str(outdir / "hero.png"),
        "left": f"splash n=96, t={water.time:.2f}, {len(water.particles)} particles",
        "right": f"taylor_green 64x64, t={sim.time:.2f}",
    }


RENDERERS = {
    "dam_break": render_dam_break,
    "splash": render_splash,
    "sloshing": render_sloshing,
    "smoke": render_smoke,
    "taylor_green": render_taylor_green,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=MEDIA)
    parser.add_argument("--only", choices=sorted({*RENDERERS, "hero"}), default=None)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    index_path = args.outdir / "showcase.json"
    summary: dict[str, Any] = {}
    if args.only is not None and index_path.exists():
        summary = json.loads(index_path.read_text())
    names = [args.only] if args.only else [*RENDERERS, "hero"]
    for name in names:
        print(f"rendering {name}", flush=True)
        start = time.perf_counter()
        summary[name] = build_hero(args.outdir) if name == "hero" else RENDERERS[name](args.outdir)
        summary[name]["wall_time"] = time.perf_counter() - start
        print(json.dumps(summary[name], indent=2, default=float), flush=True)
    index_path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
