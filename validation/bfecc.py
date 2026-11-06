"""Task 1 BFECC investigation and regression evidence.

The production bug was a swapped staggered-grid offset in the final BFECC ``v`` sweep:
it sampled with the ``u`` offsets ``(-h,-h/2)`` instead of ``(-h/2,-h)``.  This driver keeps
that legacy operation only as an explicitly labelled experimental control.  It also measures
limiter, reverse-trajectory, and correction-sign variants with every other stage unchanged.

Run ``python -m validation.bfecc``; JSON, CSV and a plot go to ``outputs/m4``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import AdvectionOptions, Scheme, _local_bounds, backtrace
from mackerel.diffuse import diffuse
from mackerel.grid import Array, BoundaryConditions, Grid, apply_velocity_bc, bilinear, pad_velocity
from mackerel.project import project_velocity
from mackerel.simulate import Simulation

OUT = Path("outputs/m4")


def bfecc_variant(
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    *,
    old_reverse: bool = False,
    limiter: bool = True,
    sign: float = 1.0,
    legacy_v_offsets: bool = False,
) -> tuple[Array, Array]:
    """BFECC with independently selectable investigation switches."""
    opts = AdvectionOptions(scheme=Scheme.BFECC)
    up, vp = pad_velocity(u, v, bc)
    h = grid.h
    Xu, Yu = grid.u_coords()
    Xv, Yv = grid.v_coords()
    xdu, ydu = backtrace(up, vp, grid, Xu, Yu, dt, opts.order)
    xdv, ydv = backtrace(up, vp, grid, Xv, Yv, dt, opts.order)
    u_new = np.asarray(bilinear(up, xdu, ydu, -h, -0.5 * h, h), dtype=np.float64)
    v_new = np.asarray(bilinear(vp, xdv, ydv, -0.5 * h, -h, h), dtype=np.float64)
    up2, vp2 = pad_velocity(u_new, v_new, bc)
    # The production implementation uses the advected field for this reverse trajectory.
    # The old-field alternative is measured separately, not assumed to be superior.
    upt, vpt = (up, vp) if old_reverse else (up2, vp2)
    uxb, uyb = backtrace(upt, vpt, grid, Xu, Yu, -dt, opts.order)
    vxb, vyb = backtrace(upt, vpt, grid, Xv, Yv, -dt, opts.order)
    u_back = np.asarray(bilinear(up2, uxb, uyb, -h, -0.5 * h, h), dtype=np.float64)
    v_back = np.asarray(bilinear(vp2, vxb, vyb, -0.5 * h, -h, h), dtype=np.float64)
    u_corrected = u + sign * 0.5 * (u - u_back)
    v_corrected = v + sign * 0.5 * (v - v_back)
    if limiter:
        u_corrected = np.clip(
            u_corrected, *_local_bounds(u, 1, bc.is_x_periodic, bc.is_y_periodic)
        )
        v_corrected = np.clip(
            v_corrected, *_local_bounds(v, 1, bc.is_x_periodic, bc.is_y_periodic)
        )
    upc, vpc = pad_velocity(u_corrected, v_corrected, bc)
    u_corr = np.asarray(bilinear(upc, xdu, ydu, -h, -0.5 * h, h), dtype=np.float64)
    # This is the corrected v-face offset.  The legacy control deliberately uses the u one.
    vx0, vy0 = (-h, -0.5 * h) if legacy_v_offsets else (-0.5 * h, -h)
    v_corr = np.asarray(bilinear(vpc, xdv, ydv, vx0, vy0, h), dtype=np.float64)
    if limiter:
        u_corr = np.clip(
            u_corr, *_local_bounds(u_corrected, 1, bc.is_x_periodic, bc.is_y_periodic)
        )
        v_corr = np.clip(
            v_corr, *_local_bounds(v_corrected, 1, bc.is_x_periodic, bc.is_y_periodic)
        )
    return u_corr, v_corr


def run_case(
    *,
    old_reverse: bool = False,
    limiter: bool = True,
    sign: float = 1.0,
    legacy_v_offsets: bool = False,
    n: int = 64,
    nu: float = 0.01,
    dt: float = 0.0025,
    end_time: float = 0.2,
) -> dict[str, Any]:
    """Run the complete split step with the selected BFECC advection experiment."""
    grid = Grid(n, n)
    sim = Simulation.taylor_green(
        grid, nu=nu, dt=dt, scheme=Scheme.BFECC, pressure_tol=1e-9, pressure_maxiter=500
    )
    e0 = sim.energy()
    divergences: list[float] = []
    steps = round(end_time / dt)
    for _ in range(steps):
        sim.u, sim.v = bfecc_variant(
            sim.u, sim.v, grid, sim.bc, dt,
            old_reverse=old_reverse, limiter=limiter, sign=sign,
            legacy_v_offsets=legacy_v_offsets,
        )
        apply_velocity_bc(sim.u, sim.v, sim.bc)
        if nu > 0.0:
            res = diffuse(sim.u, sim.v, grid, sim.bc, nu, dt, tol=1e-10, maxiter=400)
            sim.u, sim.v = res.u, res.v
        proj = project_velocity(
            sim.u, sim.v, grid, sim.bc, dt,
            system=sim._system_for(), tol=1e-9, maxiter=500,
        )
        sim.u, sim.v = proj.u, proj.v
        apply_velocity_bc(sim.u, sim.v, sim.bc)
        divergences.append(proj.div_max)
    final_time = steps * dt
    exact = float(e0 * np.exp(-4.0 * (2.0 * np.pi) ** 2 * nu * final_time))
    energy = sim.energy()
    return {
        "n": n,
        "nu": nu,
        "dt": dt,
        "time": final_time,
        "energy": energy,
        "exact_energy": exact,
        "relative_error": energy / exact - 1.0,
        "max_divergence": max(divergences),
        "old_reverse": old_reverse,
        "limiter": limiter,
        "sign": sign,
        "legacy_v_offsets": legacy_v_offsets,
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, dict[str, Any]]] = [
        ("legacy_v_offsets", {"legacy_v_offsets": True}),
        ("corrected", {}),
        ("corrected_unlimited", {"limiter": False}),
        ("corrected_old_reverse", {"old_reverse": True}),
        ("corrected_wrong_sign", {"sign": -1.0}),
    ]
    rows = [{"case": name, **run_case(**kwargs)} for name, kwargs in cases]
    (outdir / "bfecc_investigation.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (outdir / "bfecc_investigation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    names = [r["case"] for r in rows]
    values = [100.0 * r["relative_error"] for r in rows]
    colors = ["tab:red" if r["legacy_v_offsets"] else "tab:blue" for r in rows]
    ax.bar(names, values, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("relative energy error at t=0.2 (%)")
    ax.set_title("BFECC Taylor-Green investigation (64²)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(outdir / "bfecc_investigation.png", dpi=150)
    plt.close(fig)
    for row in rows:
        print(f"{row['case']:24s} E={row['energy']:.9f} rel={row['relative_error']:+.6e}")
    print(f"artifacts written to {outdir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
