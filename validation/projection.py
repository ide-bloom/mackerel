"""Milestone 3 validation: solver iteration counts, wall time and divergence reduction.

This is where the claim "multigrid is near O(n)" has to be paid for with measurements.  For
each solver and grid size we project the same random velocity field and record:

* iterations to reach the relative residual tolerance,
* wall time for the whole projection,
* the divergence left behind (max and L2).

Figures: ``outputs/m3/solver_iterations.png`` (iterations vs n, log-log),
``outputs/m3/solver_time.png`` (time vs n, log-log, with a fitted slope: 2.0 means the work
grows with the number of cells, i.e. linear in the problem size) and
``outputs/m3/divergence_reduction.png`` (divergence before/after per grid).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mackerel.grid import Grid, apply_velocity_bc, divergence, walls_bc
from mackerel.project import project_velocity

OUT = Path("outputs/m3")
SOLVER_NAMES = ["jacobi", "gauss_seidel", "conjugate_gradient", "multigrid"]


def measure(n: int, solver: str, tol: float, maxiter: int, seed: int = 0) -> dict[str, float]:
    """Project a random field on an ``n x n`` grid and report cost and accuracy."""
    g = Grid(n, n, lx=1.0)
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, g.shape_u)
    v = rng.uniform(-1.0, 1.0, g.shape_v)
    bc = walls_bc()
    apply_velocity_bc(u, v, bc)
    before = float(np.max(np.abs(divergence(u, v, g.h))))
    t0 = time.perf_counter()
    res = project_velocity(u, v, g, bc, dt=0.1, solver=solver, tol=tol, maxiter=maxiter)
    wall = time.perf_counter() - t0
    return {
        "n": float(n),
        "cells": float(n * n),
        "iterations": float(res.iterations),
        "wall_time": wall,
        "div_before": before,
        "div_after_max": res.div_max,
        "div_after_l2": res.div_l2,
        "residual": res.residual,
        "hit_maxiter": float(res.iterations >= maxiter),
    }


def run_sweep() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    budgets = {"jacobi": 4000, "gauss_seidel": 4000, "conjugate_gradient": 3000,
               "multigrid": 50}
    for n in (32, 64, 128, 256, 512):
        for name in SOLVER_NAMES:
            row: dict[str, float | str] = dict(measure(n, name, tol=1e-8, maxiter=budgets[name]))
            row["name"] = name
            rows.append(row)
            print(
                f"  n={n:4d} {name:18s} it={int(row['iterations']):5d} "
                f"t={float(row['wall_time'])*1e3:8.1f} ms  div {float(row['div_before']):.1f} -> "
                f"{float(row['div_after_max']):.2e}"
                + ("   (HIT ITERATION CAP)" if row["hit_maxiter"] else "")
            )
    return rows


def _plots(rows: list[dict[str, float | str]], outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    for name in SOLVER_NAMES:
        sel = [r for r in rows if r["name"] == name]
        ns = [float(r["n"]) for r in sel]
        its = [float(r["iterations"]) for r in sel]
        ax.loglog(ns, its, "o-", label=name)
    ax.set_xlabel("grid size n (n x n)")
    ax.set_ylabel("iterations to rel. residual 1e-8")
    ax.set_title("Pressure solve: iterations vs grid size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(outdir / "solver_iterations.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    for name in SOLVER_NAMES:
        sel = [r for r in rows if r["name"] == name]
        cells = np.array([float(r["cells"]) for r in sel])
        t = np.array([float(r["wall_time"]) for r in sel])
        slope = float(np.polyfit(np.log(cells), np.log(t), 1)[0])
        ax.loglog(cells, t * 1e3, "o-", label=f"{name} (slope {slope:.2f})")
    ax.set_xlabel("unknowns (n x n)")
    ax.set_ylabel("wall time per projection [ms]")
    ax.set_title("Pressure solve: wall time vs problem size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(outdir / "solver_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0), layout="constrained")
    ns = sorted({int(r["n"]) for r in rows})
    before = [next(r["div_before"] for r in rows if r["n"] == n) for n in ns]
    after = [max(r["div_after_max"] for r in rows if r["n"] == n) for n in ns]
    x = np.arange(len(ns), dtype=float)
    ax.bar(x - 0.2, before, width=0.4, label="max |div| before")
    ax.bar(x + 0.2, after, width=0.4, label="max |div| after (worst solver)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("grid size n")
    ax.set_title("Divergence removed by projection")
    ax.legend(fontsize=8)
    fig.savefig(outdir / "divergence_reduction.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    print("projection sweep (random field, no-slip walls, tol 1e-8)")
    rows = run_sweep()
    _plots(rows, outdir)

    print("\niteration counts")
    for name in SOLVER_NAMES:
        sel = [r for r in rows if r["name"] == name]
        print(f"  {name:18s} " + " ".join(
            f"n={int(float(r['n']))}:{int(float(r['iterations']))}"
            + ("*" if r["hit_maxiter"] else "")
            for r in sel))
    print("\nwall time slopes (vs number of unknowns; 1.0 = linear in problem size)")
    for name in SOLVER_NAMES:
        sel = [r for r in rows if r["name"] == name]
        cells = np.array([float(r["cells"]) for r in sel])
        t = np.array([float(r["wall_time"]) for r in sel])
        print(f"  {name:18s} slope {float(np.polyfit(np.log(cells), np.log(t), 1)[0]):.2f}")
    print("\n(* = iteration cap reached, result not fully converged)")
    print(f"\nfigures written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
