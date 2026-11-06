"""Milestone 5: lid-driven cavity against Ghia, Ghia & Shin (1982).

The top wall moves at unit speed and the other walls are fixed.  For each Reynolds number we
march the full MAC Navier--Stokes solver to a steady state, interpolate cell-centred velocities
to the tabulated centreline stations, and report max/RMS deviations.  Reference values are in
``validation/data/ghia_1982.csv`` with the citation and transcription notes alongside them.

``python -m validation.lid_cavity [output_dir]`` writes figures and raw results to
``outputs/m5``.  The optional ``MACKEREL_CAVITY_N`` environment variable changes the default
grid for a cheaper smoke run; it does not alter the reference data.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme
from mackerel.grid import Array, Grid, walls_bc
from mackerel.simulate import Simulation

OUT = Path("outputs/m5")
DATA = Path(__file__).with_name("data") / "ghia_1982.csv"


def reference(reynolds: int) -> dict[str, tuple[Array, Array]]:
    """Load ``(coordinates, values)`` for the two Ghia centreline tables."""
    rows: dict[str, list[tuple[float, float]]] = {
        "u_vertical_centerline": [],
        "v_horizontal_centerline": [],
    }
    with DATA.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["re"]) == reynolds:
                rows[row["component"]].append((float(row["coordinate"]), float(row["value"])))
    return {
        key: (
            np.asarray([p[0] for p in values], dtype=np.float64),
            np.asarray([p[1] for p in values], dtype=np.float64),
        )
        for key, values in rows.items()
    }


def _with_walls(x: Array, y: Array, left: float, right: float) -> tuple[Array, Array]:
    return np.r_[0.0, x, 1.0], np.r_[left, y, right]


def centreline_profiles(sim: Simulation) -> tuple[Array, Array, Array, Array]:
    """Return ``(y, u(x=.5,y), x, v(x,y=.5))`` with exact wall endpoint values."""
    g = sim.grid
    uc = g.u_to_centers(sim.u)
    vc = g.v_to_centers(sim.v)
    # Average the two centre-adjacent cell columns/rows for an even grid.  np.interp then
    # supplies the remaining (off-grid) interpolation; wall values are physical BC data.
    mid = g.nx // 2
    u_profile = np.mean(uc[:, mid - 1 : mid + 1], axis=1)
    y_u, u_profile = _with_walls(g.y_centers(), u_profile, 0.0, 1.0)
    mid_y = g.ny // 2
    v_profile = np.mean(vc[mid_y - 1 : mid_y, :], axis=0)
    x_v, v_profile = _with_walls(g.x_centers(), v_profile, 0.0, 0.0)
    return y_u, u_profile, x_v, v_profile


def run_cavity(
    reynolds: int,
    n: int = 48,
    dt: float = 0.01,
    max_time: float = 60.0,
) -> dict[str, Any]:
    """Run one case until the maximum velocity change in 100 steps is below 2e-8."""
    grid = Grid(n, n, lx=1.0)
    sim = Simulation.at_rest(
        grid,
        walls_bc((1.0, 0.0)),
        nu=1.0 / reynolds,
        dt=dt,
        scheme=Scheme.MACCORMACK,
        limiter=True,
        pressure_tol=1e-8,
        pressure_maxiter=2000,
    )
    previous = sim.u.copy()
    t0 = time.perf_counter()
    change = float("nan")
    converged = False
    pressure_iterations_max = 0
    pressure_residual_max = 0.0
    while sim.time < max_time - 1e-12:
        infos = sim.run_steps(100)
        pressure_iterations_max = max(pressure_iterations_max, max(i.pressure_iterations for i in infos))
        pressure_residual_max = max(pressure_residual_max, max(i.pressure_residual for i in infos))
        change = float(np.max(np.abs(sim.u - previous)))
        if change < 2e-8:
            converged = True
            break
        previous = sim.u.copy()
    y_u, u_line, x_v, v_line = centreline_profiles(sim)
    ref = reference(reynolds)
    y_ref, u_ref = ref["u_vertical_centerline"]
    x_ref, v_ref = ref["v_horizontal_centerline"]
    u_at_ref = np.interp(y_ref, y_u, u_line)
    v_at_ref = np.interp(x_ref, x_v, v_line)
    eu = u_at_ref - u_ref
    ev = v_at_ref - v_ref
    # Ghia's printed Re=400 v(x=.9063)=-.23827 is a known isolated tabulation outlier.
    # Keep it in the data and in the *_all metrics, but exclude exactly that one point from
    # the primary comparison so it does not dominate the solver error.
    primary_u = np.ones(y_ref.shape, dtype=bool)
    primary_v = np.ones(x_ref.shape, dtype=bool)
    if reynolds == 400:
        primary_v &= ~np.isclose(x_ref, 0.9063)
    return {
        "re": reynolds,
        "n": n,
        "dt": dt,
        "time": float(sim.time),
        "steps": len(sim.history),
        "wall_time": time.perf_counter() - t0,
        "max_time": max_time,
        "converged": converged,
        "steady_change": float(change),
        "pressure_maxiter": sim.config.pressure_maxiter,
        "pressure_iterations_max": pressure_iterations_max,
        "pressure_residual_max": pressure_residual_max,
        "energy": sim.energy(),
        "max_divergence": float(max((d.div_max for d in sim.history), default=0.0)),
        "u_max_error": float(np.max(np.abs(eu[primary_u]))),
        "u_rms_error": float(np.sqrt(np.mean(eu[primary_u] ** 2))),
        "v_max_error": float(np.max(np.abs(ev[primary_v]))),
        "v_rms_error": float(np.sqrt(np.mean(ev[primary_v] ** 2))),
        "u_max_error_all": float(np.max(np.abs(eu))),
        "u_rms_error_all": float(np.sqrt(np.mean(eu**2))),
        "v_max_error_all": float(np.max(np.abs(ev))),
        "v_rms_error_all": float(np.sqrt(np.mean(ev**2))),
        "y_reference": y_ref,
        "u_reference": u_ref,
        "u_numerical": u_at_ref,
        "x_reference": x_ref,
        "v_reference": v_ref,
        "v_numerical": v_at_ref,
        "y_line": y_u,
        "u_line": u_line,
        "x_line": x_v,
        "v_line": v_line,
        "uc": grid.u_to_centers(sim.u),
        "vc": grid.v_to_centers(sim.v),
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    n = int(os.environ.get("MACKEREL_CAVITY_N", "48"))
    rows = [run_cavity(re, n=n) for re in (100, 400)]
    serial = []
    for row in rows:
        serial.append({k: v for k, v in row.items() if not isinstance(v, np.ndarray)})
    (outdir / "cavity.json").write_text(json.dumps(serial, indent=2) + "\n")
    arrays: dict[str, Any] = {
        f"re{row['re']}_{key}": value
        for row in rows for key, value in row.items() if isinstance(value, np.ndarray)
    }
    np.savez(outdir / "cavity_profiles.npz", **arrays)
    with (outdir / "cavity.csv").open("w", newline="") as f:
        fields = [k for k in serial[0] if not k.endswith("reference")]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in serial])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for r, row in enumerate(rows):
        axes[r, 0].plot(row["u_line"], row["y_line"], "b-", label=f"mackerel n={row['n']}")
        axes[r, 0].plot(row["u_reference"], row["y_reference"], "ro", ms=4, label="Ghia et al.")
        axes[r, 0].set_title(f"u at x=0.5, Re={row['re']}; max err {row['u_max_error']:.3f}")
        axes[r, 0].set_xlabel("u/U_lid")
        axes[r, 0].set_ylabel("y")
        axes[r, 1].plot(row["x_line"], row["v_line"], "b-", label=f"mackerel n={row['n']}")
        axes[r, 1].plot(row["v_reference"], row["x_reference"], "ro", ms=4, label="Ghia et al.")
        axes[r, 1].set_title(f"v at y=0.5, Re={row['re']}; max err {row['v_max_error']:.3f}")
        axes[r, 1].set_xlabel("x")
        axes[r, 1].set_ylabel("v/U_lid")
        for ax in axes[r]:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.savefig(outdir / "cavity_ghia_profiles.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, row in zip(axes, rows, strict=True):
        speed = np.sqrt(row["uc"] ** 2 + row["vc"] ** 2)
        im = ax.imshow(speed, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        skip = max(1, row["n"] // 16)
        x = np.linspace(0, 1, row["n"], endpoint=False) + 0.5 / row["n"]
        X, Y = np.meshgrid(x, x)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  row["uc"][::skip, ::skip], row["vc"][::skip, ::skip], color="white")
        ax.set_title(f"Re={row['re']}, t={row['time']:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="speed")
    fig.savefig(outdir / "cavity_velocity.png", dpi=150)
    plt.close(fig)

    print("Lid-driven cavity / Ghia et al. (1982)")
    for row in rows:
        print(
            f"  Re={row['re']:4d} n={row['n']} t={row['time']:.2f} "
            f"wall={row['wall_time']:.1f}s "
            f"u max/RMS={row['u_max_error']:.4f}/{row['u_rms_error']:.4f} "
            f"v max/RMS={row['v_max_error']:.4f}/{row['v_rms_error']:.4f} "
            f"(all points {row['v_max_error_all']:.4f}) div={row['max_divergence']:.2e}"
        )
    print(f"figures and data written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
