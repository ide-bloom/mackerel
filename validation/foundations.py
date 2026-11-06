"""Milestone 1 validation: grid geometry, interpolation order, energy/enstrophy order.

Produces the figures that back ``docs/MATH.md`` and ``docs/milestones/M1.md``:

* ``outputs/m1/checkerboard.png`` -- the ``(-1)^(i+j)`` mode under collocated vs staggered
  gradients (the numerical demonstration that staggering has no pressure null space),
* ``outputs/m1/convergence.png`` -- measured order of accuracy of the sampled Taylor-Green
  energy, enstrophy and velocity, plus bilinear interpolation error.

Run with ``mackerel validate foundations`` (or ``python -m validation.foundations``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mackerel.analytic import (
    checkerboard_pressure_field,
    taylor_green,
    taylor_green_enstrophy,
    taylor_green_velocity,
)
from mackerel.diagnostics import (
    divergence_field,
    enstrophy,
    kinetic_energy,
    vorticity_field,
)
from mackerel.grid import Grid, sample_cell

OUT = Path("outputs/m1")


def checkerboard_measurements(n: int = 16) -> dict[str, float]:
    """Max gradient produced by ``p = (-1)^(i+j)`` on a collocated vs a staggered grid."""
    g = Grid(n, n, lx=1.0)
    p = checkerboard_pressure_field(g, amplitude=1.0)
    collocated = np.zeros_like(p)
    collocated[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * g.h)
    staggered = np.zeros((g.ny, g.nx + 1))
    staggered[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / g.h
    return {
        "h": g.h,
        "collocated_max_grad": float(np.max(np.abs(collocated))),
        "staggered_max_grad": float(np.max(np.abs(staggered))),
        "staggered_expected": 2.0 / g.h,
    }


def _bisection_order(h: list[float], e: list[float]) -> float:
    return float(np.polyfit(np.log(h), np.log(np.maximum(e, 1e-300)), 1)[0])


def convergence_study() -> dict[str, list[float]]:
    """Measure the convergence rate of energy, enstrophy and velocity error.

    The reference values are the exact integrals of the analytic field over the unit
    square, not a finer-grid simulation, so what is measured is a property of the
    *sampling and quadrature* operators alone.
    """
    sizes = [8, 16, 32, 64, 128]
    h: list[float] = []
    e_energy: list[float] = []
    e_enstrophy: list[float] = []
    e_velocity: list[float] = []
    e_interp: list[float] = []
    rng = np.random.default_rng(0)

    # exact values for the unit-square Taylor-Green field at t = 0
    exact_energy = 0.25
    exact_enstrophy = taylor_green_enstrophy(0.0, 0.0)  # 2 k^2 E0 = 2 pi^2

    for n in sizes:
        g = Grid(n, n, lx=1.0)
        u, v = taylor_green(g)
        h.append(g.h)
        e_energy.append(abs(kinetic_energy(u, v, g) - exact_energy))
        e_enstrophy.append(abs(enstrophy(u, v, g) - exact_enstrophy))

        Xu, Yu = g.u_coords()
        ue, _ = taylor_green_velocity(g, Xu, Yu)
        Xv, Yv = g.v_coords()
        _, ve = taylor_green_velocity(g, Xv, Yv)
        e_velocity.append(float(np.sqrt(np.mean((u - ue) ** 2) + np.mean((v - ve) ** 2))))

        X, Y = g.cell_coords()
        field = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        xs = rng.uniform(0.0, 1.0, 20000)
        ys = rng.uniform(0.0, 1.0, 20000)
        got = sample_cell(field, g, xs, ys, clamp=False)
        want = np.sin(2 * np.pi * xs) * np.sin(2 * np.pi * ys)
        e_interp.append(float(np.sqrt(np.mean((got - want) ** 2))))

    return {
        "h": h,
        "energy": e_energy,
        "enstrophy": e_enstrophy,
        "velocity": e_velocity,
        "interpolation": e_interp,
    }


def _plot(check: dict[str, float], conv: dict[str, list[float]], outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # --- figure 1: the checkerboard mode
    g = Grid(16, 16, lx=1.0)
    p = checkerboard_pressure_field(g)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    im = axes[0].imshow(p, origin="lower", cmap="RdBu_r")
    axes[0].set_title(f"p = (-1)^(i+j)  (h = {g.h:g})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for ax, key, title in (
        (axes[1], "collocated_max_grad", "collocated centred gradient"),
        (axes[2], "staggered_max_grad", "staggered face gradient"),
    ):
        ax.bar(["max |grad p|"], [check[key]], color="#c0392b" if "collocated" in key else "#27ae60")
        ax.set_title(f"{title}\n{check[key]:.3g}")
        ax.set_ylim(0, 1.15 * max(check["staggered_max_grad"], 1e-9))
    fig.suptitle("The checkerboard mode is invisible to a collocated grid, loud on a staggered one")
    fig.tight_layout()
    fig.savefig(outdir / "checkerboard.png", dpi=140)
    plt.close(fig)

    # --- figure 2: convergence
    h = np.array(conv["h"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, style in (
        ("energy", "|E_h - E_exact|", "o-"),
        ("enstrophy", "|Z_h - Z_exact|", "s-"),
        ("velocity", "RMS velocity error", "^-"),
        ("interpolation", "bilinear interpolation RMS", "d--"),
    ):
        e = np.array(conv[key])
        order = _bisection_order(list(h), list(e))
        ax.loglog(h, e, style, label=f"{label}  (order {order:.2f})")
    ax.loglog(h, 3.3 * h**2, "k:", lw=1, label="reference slope h^2")
    ax.set_xlabel("h")
    ax.set_ylabel("error")
    ax.set_title("Milestone 1: second-order sampling and quadrature")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "convergence.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the milestone 1 validation and print the measured numbers."""
    outdir = Path(argv[0]) if argv else OUT
    check = checkerboard_measurements()
    conv = convergence_study()
    _plot(check, conv, outdir)

    print("checkerboard (collocated vs staggered)")
    print(f"  h                        {check['h']:.4g}")
    print(f"  collocated max |grad p|  {check['collocated_max_grad']:.3e}")
    print(f"  staggered  max |grad p|  {check['staggered_max_grad']:.6f} "
          f"(= 2/h = {check['staggered_expected']:.6f})")

    print("\nconvergence study")
    h = conv["h"]
    for name in ("energy", "enstrophy", "velocity", "interpolation"):
        e = conv[name]
        print(f"  {name:14s} order {_bisection_order(h, e):.3f}   errors "
              + " ".join(f"{x:.3e}" for x in e))

    # divergence diagnostics for the analytic field sampled on the grid
    for n in (16, 64):
        g = Grid(n, n, lx=1.0)
        u, v = taylor_green(g)
        print(f"  max |div| of corner-curl TG field on {n}^2: "
              f"{np.max(np.abs(divergence_field(u, v, g))):.3e}")
        print(f"  vorticity range on {n}^2: "
              f"[{vorticity_field(u, v, g).min():.3f}, {vorticity_field(u, v, g).max():.3f}]")
    print(f"\nfigures written to {outdir}/checkerboard.png and {outdir}/convergence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
