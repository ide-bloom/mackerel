"""Milestone 6 moving-obstacle demonstration.

A circular solid translates through initially quiescent fluid in a closed box.  The cell mask
is rebuilt at each time level and the pressure hierarchy is invalidated, while every solid
face keeps the obstacle's translational velocity.  The output is a genuine time-stepped
sequence, not a translated post-processing image.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.grid import Grid, walls_bc
from mackerel.obstacles import MovingObstacle, solid_face_masks
from mackerel.render import RGBArray, save_frames_gif, save_image
from mackerel.simulate import Simulation

OUT = Path("outputs/m6")


def _speed_frame(sim: Simulation) -> RGBArray:
    import matplotlib

    uc = sim.grid.u_to_centers(sim.u)
    vc = sim.grid.v_to_centers(sim.v)
    speed = np.sqrt(uc * uc + vc * vc)
    norm = np.clip(speed / 0.35, 0.0, 1.0)
    rgb = matplotlib.colormaps["viridis"](norm)[:, :, :3]
    if sim.solid is not None:
        rgb[sim.solid] = (0.08, 0.08, 0.08)
    return np.ascontiguousarray((np.flipud(rgb) * 255).astype(np.uint8))


def run_moving(n: int = 32, end_time: float = 2.0, dt: float = 0.005) -> dict[str, Any]:
    grid = Grid(2 * n, n, lx=2.0)
    obstacle = MovingObstacle("circle", (0.35, 0.5), 0.12, velocity=(0.15, 0.0))
    sim = Simulation.at_rest(grid, walls_bc(), nu=0.002, dt=dt, pressure_tol=1e-8)
    frames: list[RGBArray] = []
    centres: list[float] = []
    max_div = 0.0
    t0 = time.perf_counter()
    steps = round(end_time / dt)
    for k in range(steps):
        mask = obstacle.mask(grid, sim.time)
        sim.set_solid(mask, obstacle.velocity)
        info = sim.step()
        max_div = max(max_div, info.div_max)
        centres.append(obstacle.center(sim.time)[0])
        if k % 20 == 0 or k == steps - 1:
            frames.append(_speed_frame(sim))
    um, vm = solid_face_masks(sim.solid) if sim.solid is not None else (None, None)
    face_error = 0.0
    if um is not None and vm is not None:
        face_error = max(
            float(np.max(np.abs(sim.u[um] - obstacle.velocity[0]))),
            float(np.max(np.abs(sim.v[vm] - obstacle.velocity[1]))),
        )
    return {
        "n": n,
        "time": float(sim.time),
        "steps": steps,
        "wall_time": time.perf_counter() - t0,
        "obstacle_start_x": obstacle.center0[0],
        "obstacle_final_x": obstacle.center(sim.time)[0],
        "expected_final_x": obstacle.center0[0] + obstacle.velocity[0] * sim.time,
        "solid_face_velocity_error": face_error,
        "max_divergence": max_div,
        "energy": sim.energy(),
        "centres": np.asarray(centres),
        "frames": frames,
        "speed": np.sqrt(grid.u_to_centers(sim.u) ** 2 + grid.v_to_centers(sim.v) ** 2),
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_moving()
    scalar = {k: v for k, v in result.items() if not isinstance(v, (np.ndarray, list))}
    (outdir / "moving.json").write_text(json.dumps(scalar, indent=2) + "\n")
    save_image(result["frames"][-1], outdir / "moving_obstacle.png")
    save_frames_gif(result["frames"], outdir / "moving_obstacle.gif", duration_ms=80)
    print("Moving obstacle")
    for key, value in scalar.items():
        print(f"  {key}: {value}")
    print(f"figures written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
