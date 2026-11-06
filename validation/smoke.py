"""Milestone 6 buoyant smoke-plume demonstration.

A hot patch is continuously replenished at the bottom of a closed-sided box with an open
outflow at the top.  Temperature is advected and diffused independently of velocity viscosity;
its anomaly drives upward buoyancy.  The script reports the scalar centre of mass and writes a
plume figure plus a GIF of the actual time steps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme
from mackerel.grid import BoundaryConditions, Grid, Kind, SideBC
from mackerel.render import RGBArray, save_frames_gif, save_image
from mackerel.simulate import Simulation, StepInfo

OUT = Path("outputs/m6")


def plume_bc() -> BoundaryConditions:
    wall = SideBC(Kind.NO_SLIP)
    return BoundaryConditions(wall, wall, wall, SideBC(Kind.OUTFLOW))


def _rgb(field: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> RGBArray:
    import matplotlib

    norm = np.clip((field - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    rgb = matplotlib.colormaps["inferno"](norm)[:, :, :3]
    return np.ascontiguousarray((np.flipud(rgb) * 255).astype(np.uint8))


def run_plume(n: int = 48, end_time: float = 4.0, dt: float = 0.005) -> dict[str, Any]:
    grid = Grid(n, n, lx=1.0)
    sim = Simulation.at_rest(
        grid,
        plume_bc(),
        nu=1e-3,
        dt=dt,
        scalar_diffusivity=2e-4,
        gravity=9.81,
        scalar_buoyancy=0.2,
        scheme=Scheme.SEMI_LAGRANGIAN,
        pressure_tol=1e-8,
        pressure_maxiter=1500,
    )
    sim.add_scalar("temperature", 0.0)
    X, Y = grid.cell_coords()
    source = (X - 0.5) ** 2 + (Y - 0.08) ** 2 < 0.065**2
    frames: list[RGBArray] = []
    times: list[float] = []
    masses: list[float] = []
    centres: list[float] = []

    def replenish() -> None:
        sim.scalars["temperature"][source] = 1.0

    def record(state: Simulation, _info: StepInfo) -> None:
        temperature = state.scalars["temperature"]
        mass = float(np.sum(temperature) * grid.cell_area)
        centre = float(np.sum(temperature * Y) / max(np.sum(temperature), 1e-30))
        times.append(state.time)
        masses.append(mass)
        centres.append(centre)
        if len(frames) < 80 and (len(times) == 1 or state.time - times[-2] >= 0.049):
            frames.append(_rgb(temperature))

    replenish()
    t0 = time.perf_counter()
    steps = round(end_time / dt)
    for _ in range(steps):
        replenish()
        sim.step()
        record(sim, StepInfo(sim.time, dt, 0.0, 0, 0.0, 0.0, 0.0))
    return {
        "n": n,
        "dt": dt,
        "time": float(sim.time),
        "steps": steps,
        "wall_time": time.perf_counter() - t0,
        "initial_centre_y": centres[0],
        "final_centre_y": centres[-1],
        "max_temperature": float(np.max(sim.scalars["temperature"])),
        "min_temperature": float(np.min(sim.scalars["temperature"])),
        "final_divergence": float(sim.history[-1].div_max),
        "max_divergence": float(max(d.div_max for d in sim.history)),
        "times": np.asarray(times),
        "mass": np.asarray(masses),
        "centre_y": np.asarray(centres),
        "temperature": sim.scalars["temperature"],
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_plume()
    scalar = {k: v for k, v in result.items() if not isinstance(v, (np.ndarray, list))}
    (outdir / "smoke.json").write_text(json.dumps(scalar, indent=2) + "\n")
    np.savez(
        outdir / "smoke_fields.npz",
        times=result["times"], mass=result["mass"], centre_y=result["centre_y"],
        temperature=result["temperature"],
    )
    save_image(result["frames"][-1], outdir / "smoke_plume.png")
    save_frames_gif(result["frames"], outdir / "smoke_plume.gif", duration_ms=70)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    im = axes[0].imshow(result["temperature"], origin="lower", extent=(0, 1, 0, 1),
                        cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"Buoyant temperature, t={result['time']:.1f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0], label="T")
    axes[1].plot(result["times"], result["centre_y"], label="scalar centre of mass")
    axes[1].plot(result["times"], result["mass"], label="scalar integral")
    axes[1].set_xlabel("time")
    axes[1].set_title("Plume transport diagnostics")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(outdir / "smoke_diagnostics.png", dpi=150)
    plt.close(fig)
    print("Smoke plume")
    for key, value in scalar.items():
        print(f"  {key}: {value}")
    print(f"figures and data written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
