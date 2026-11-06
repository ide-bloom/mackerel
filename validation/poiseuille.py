"""Milestone 4 plane-Poiseuille validation.

A body force ``f`` in a periodic-x channel with no-slip top and bottom walls has the steady
solution ``u(y) = f y (H-y) / (2 nu)``, ``v = 0``.  The complete time-stepping solver is
marched until the velocity stops changing, then compared with that profile at cell centres.
The refinement table reports the measured L2 error and its observed order.

Run ``python -m validation.poiseuille`` (optionally pass an output directory).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.grid import BoundaryConditions, Grid, Kind, SideBC
from mackerel.simulate import Simulation

OUT = Path("outputs/m4")


def channel_bc() -> BoundaryConditions:
    p = SideBC(Kind.PERIODIC)
    wall = SideBC(Kind.NO_SLIP)
    return BoundaryConditions(p, p, wall, wall)


def steady_run(
    n: int,
    force: float = 0.1,
    nu: float = 0.05,
    dt: float = 0.02,
    max_time: float = 120.0,
) -> dict[str, Any]:
    grid = Grid(n, n, lx=1.0)
    sim = Simulation.at_rest(
        grid,
        channel_bc(),
        nu=nu,
        dt=dt,
        scheme="semi_lagrangian",
        pressure_tol=1e-10,
        buoyancy=(force, 0.0),
    )
    previous = sim.u.copy()
    steps = 0
    while sim.time < max_time - 1e-12:
        sim.run_steps(50)
        steps += 50
        change = float(np.max(np.abs(sim.u - previous)))
        if change < 1e-9:
            break
        previous = sim.u.copy()
    uc = grid.u_to_centers(sim.u)
    y = grid.y_centers()
    exact = (force / (2.0 * nu) * y * (grid.ly - y))[:, None] * np.ones((1, n))
    error = uc - exact
    l2 = float(np.sqrt(np.mean(error**2)))
    return {
        "n": n,
        "time": float(sim.time),
        "steps": steps,
        "dt": dt,
        "force": force,
        "nu": nu,
        "l2_error": l2,
        "relative_l2_error": l2 / float(np.sqrt(np.mean(exact**2))),
        "max_u": float(np.max(uc)),
        "exact_max_u": float(force * grid.ly**2 / (8.0 * nu)),
        "max_v": float(np.max(np.abs(sim.v))),
        "max_divergence": float(max((d.div_max for d in sim.history), default=0.0)),
        "y": y,
        "u": uc[:, 0],
        "exact": exact[:, 0],
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [steady_run(n) for n in (16, 32, 64)]
    errors = np.array([r["l2_error"] for r in rows], dtype=float)
    ns = np.array([r["n"] for r in rows], dtype=float)
    orders = [float("nan")] + [
        float(np.log(errors[i - 1] / errors[i]) / np.log(ns[i] / ns[i - 1]))
        for i in range(1, len(rows))
    ]
    for row, order in zip(rows, orders, strict=True):
        row["observed_order"] = order

    serial = [{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} for row in rows]
    (outdir / "poiseuille.json").write_text(json.dumps(serial, indent=2) + "\n")
    with (outdir / "poiseuille.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.8), layout="constrained")
    for row in rows:
        ax.plot(row["u"], row["y"], "o-", ms=3, label=f"n={row['n']}")
    ax.plot(rows[-1]["exact"], rows[-1]["y"], "k--", lw=2, label="analytic")
    ax.set_xlabel("streamwise velocity u")
    ax.set_ylabel("y")
    ax.set_title("Steady plane Poiseuille profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "poiseuille_profiles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), layout="constrained")
    ax.loglog(ns, errors, "o-", label="measured L2 error")
    ax.loglog(ns, errors[0] * (ns[0] / ns) ** 2, "k:", label="second-order reference")
    ax.set_xlabel("grid n")
    ax.set_ylabel("L2 velocity error")
    ax.set_title("Poiseuille spatial refinement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "poiseuille_convergence.png", dpi=150)
    plt.close(fig)

    print("Plane Poiseuille steady-state validation")
    for row in rows:
        print(
            f"  {row['n']:3d}² t={row['time']:7.2f} L2={row['l2_error']:.4e} "
            f"rel={row['relative_l2_error']:.3e} order={row['observed_order']:.2f} "
            f"umax={row['max_u']:.7f} div={row['max_divergence']:.2e}"
        )
    print(f"figures and data written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
