"""Taylor--Green spatial and temporal convergence studies.

The old fixed-``dt`` energy-only sweep mixed spatial error, temporal error and the initial
MAC quadrature error.  This driver reports velocity L2 error against the analytic field as
well as energy error against the exact decay.  The spatial study uses a short final time and
a fixed small step so the temporal contribution is subdominant; the temporal study uses a
fixed 128² grid.  A symmetric Strang-style sequence is included as a measured negative
control, not silently adopted as a solver improvement.

Run ``python -m validation.convergence``; outputs go to ``outputs/m4/convergence``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme, advect_velocity
from mackerel.analytic import taylor_green_velocity
from mackerel.diffuse import diffuse
from mackerel.grid import Grid, apply_velocity_bc
from mackerel.project import project_velocity
from mackerel.simulate import Simulation

OUT = Path("outputs/m4/convergence")
K = 2.0 * np.pi


def velocity_errors(sim: Simulation, u0: np.ndarray, v0: np.ndarray) -> tuple[float, float]:
    """Return L2 errors against analytic and initially sampled decaying TG fields."""
    grid = sim.grid
    decay = np.exp(-2.0 * K**2 * sim.config.nu * sim.time)
    Xu, Yu = grid.u_coords()
    Xv, Yv = grid.v_coords()
    ua, _ = taylor_green_velocity(grid, Xu, Yu)
    _, va = taylor_green_velocity(grid, Xv, Yv)
    ua *= decay
    va *= decay
    uc = grid.u_to_centers(sim.u)
    vc = grid.v_to_centers(sim.v)
    uac = grid.u_to_centers(ua)
    vac = grid.v_to_centers(va)
    analytic = float(np.sqrt(np.sum((uc - uac) ** 2 + (vc - vac) ** 2) * grid.cell_area))
    udc = grid.u_to_centers(u0 * decay)
    vdc = grid.v_to_centers(v0 * decay)
    sampled = float(np.sqrt(np.sum((uc - udc) ** 2 + (vc - vdc) ** 2) * grid.cell_area))
    return analytic, sampled


def strang_step(sim: Simulation, dt: float) -> float:
    """Experimental symmetric P/2-D/2-A-D/2-P/2 step for force-free TG only."""
    grid = sim.grid
    cfg = sim.config
    proj = project_velocity(
        sim.u, sim.v, grid, sim.bc, dt / 2.0,
        system=sim._system_for(), tol=cfg.pressure_tol, maxiter=cfg.pressure_maxiter,
    )
    sim.u, sim.v = proj.u, proj.v
    if cfg.nu > 0.0:
        res = diffuse(sim.u, sim.v, grid, sim.bc, cfg.nu, dt / 2.0, tol=1e-10, maxiter=400)
        sim.u, sim.v = res.u, res.v
    sim.u, sim.v = advect_velocity(sim.u, sim.v, grid, sim.bc, dt, cfg.advection_options())
    apply_velocity_bc(sim.u, sim.v, sim.bc)
    if cfg.nu > 0.0:
        res = diffuse(sim.u, sim.v, grid, sim.bc, cfg.nu, dt / 2.0, tol=1e-10, maxiter=400)
        sim.u, sim.v = res.u, res.v
    proj = project_velocity(
        sim.u, sim.v, grid, sim.bc, dt / 2.0,
        system=sim._system_for(), tol=cfg.pressure_tol, maxiter=cfg.pressure_maxiter,
    )
    sim.u, sim.v = proj.u, proj.v
    apply_velocity_bc(sim.u, sim.v, sim.bc)
    sim.time += dt
    return proj.div_max


def run_case(
    n: int,
    dt: float,
    end_time: float,
    *,
    splitting: str = "lie",
    scheme: Scheme = Scheme.MACCORMACK,
    nu: float = 0.01,
) -> dict[str, Any]:
    """Run one fixed-grid/fixed-step case and retain all error measures."""
    if splitting not in {"lie", "strang"}:
        raise ValueError("splitting must be 'lie' or 'strang'")
    grid = Grid(n, n)
    sim = Simulation.taylor_green(
        grid, nu=nu, dt=dt, scheme=scheme, pressure_tol=1e-9, pressure_maxiter=500
    )
    u0, v0 = sim.u.copy(), sim.v.copy()
    e0 = sim.energy()
    start = time.perf_counter()
    if splitting == "lie":
        infos = sim.run_until(end_time)
        max_divergence = max(info.div_max for info in infos)
    else:
        max_divergence = 0.0
        for _ in range(round(end_time / dt)):
            max_divergence = max(max_divergence, strang_step(sim, dt))
    analytic_l2, sampled_l2 = velocity_errors(sim, u0, v0)
    exact_energy = e0 * np.exp(-4.0 * K**2 * nu * sim.time)
    return {
        "n": n,
        "dt": dt,
        "time": sim.time,
        "steps": round(end_time / dt),
        "splitting": splitting,
        "scheme": scheme.value,
        "nu": nu,
        "energy": sim.energy(),
        "exact_energy": exact_energy,
        "energy_relative_error": sim.energy() / exact_energy - 1.0,
        "velocity_l2": analytic_l2,
        "sampled_velocity_l2": sampled_l2,
        "max_divergence": max_divergence,
        "wall_time": time.perf_counter() - start,
    }


def observed_order(scales: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    """Fit ``error ~ scale**order`` and return pairwise orders as well."""
    scales = np.asarray(scales, dtype=np.float64)
    errors = np.maximum(np.abs(np.asarray(errors, dtype=np.float64)), 1e-30)
    order = float(np.polyfit(np.log(scales), np.log(errors), 1)[0])
    pairwise = np.diff(np.log(errors)) / np.diff(np.log(scales))
    return {"order": order, "pairwise": pairwise.tolist()}


def _row_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Chosen before the study: dt is 40 times smaller than the M4 baseline per unit time,
    # and the short end time keeps 800 steps per spatial case practical.
    spatial = [run_case(n, 6.25e-5, 0.05) for n in (32, 64, 128)]
    # The temporal range brackets the M4 dt=.0025 case on a fixed fine grid.
    temporal = [run_case(128, dt, 0.2) for dt in (0.004, 0.002, 0.001, 0.0005)]
    # A separate symmetric-splitting control at 64²; no forces, scalars or obstacles are
    # involved, so the experimental strang_step has exactly the same physical terms as Lie.
    strang = [run_case(64, dt, 0.2, splitting="strang") for dt in (0.008, 0.004, 0.002, 0.001)]
    lie64 = [run_case(64, dt, 0.2) for dt in (0.008, 0.004, 0.002, 0.001)]
    orders = {
        "spatial_velocity_l2": observed_order(
            np.array([1.0 / r["n"] for r in spatial]), np.array([r["velocity_l2"] for r in spatial])
        ),
        "spatial_energy_abs_relative": observed_order(
            np.array([1.0 / r["n"] for r in spatial]),
            np.array([r["energy_relative_error"] for r in spatial]),
        ),
        "temporal_energy_abs_relative": observed_order(
            np.array([r["dt"] for r in temporal]),
            np.array([r["energy_relative_error"] for r in temporal]),
        ),
        "temporal_velocity_l2": observed_order(
            np.array([r["dt"] for r in temporal]), np.array([r["velocity_l2"] for r in temporal])
        ),
        "strang_energy_abs_relative": observed_order(
            np.array([r["dt"] for r in strang]), np.array([r["energy_relative_error"] for r in strang])
        ),
        "lie64_energy_abs_relative": observed_order(
            np.array([r["dt"] for r in lie64]), np.array([r["energy_relative_error"] for r in lie64])
        ),
    }
    report = {"spatial": spatial, "temporal": temporal, "strang": strang,
              "lie64": lie64, "orders": orders}
    (args.outdir / "convergence.json").write_text(json.dumps(report, indent=2) + "\n")
    _row_csv(spatial, args.outdir / "spatial.csv")
    _row_csv(temporal, args.outdir / "temporal.csv")
    _row_csv(strang, args.outdir / "strang.csv")
    _row_csv(lie64, args.outdir / "lie64.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    ns = np.array([r["n"] for r in spatial])
    errs = np.array([r["velocity_l2"] for r in spatial])
    ax.loglog(ns, errs, "o-", label=f"measured, order {orders['spatial_velocity_l2']['order']:.2f}")
    ax.loglog(ns, errs[0] * (ns[0] / ns) ** 2, "k:", label="second-order reference")
    ax.set(xlabel="grid n", ylabel="velocity L2 error", title="Spatial study, dt=6.25e-5, t=.05")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    for rows, label in ((temporal, "Lie 128²"), (lie64, "Lie 64²"), (strang, "Strang control 64²")):
        dt = np.array([r["dt"] for r in rows])
        err = np.abs([r["energy_relative_error"] for r in rows])
        ax.loglog(dt, err, "o-", label=label)
    ax.set(xlabel="dt", ylabel="absolute relative energy error", title="Temporal study, t=.2")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(args.outdir / "convergence.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"orders": orders}, indent=2))
    print(f"artifacts written to {args.outdir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
