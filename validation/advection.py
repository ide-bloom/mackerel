"""Milestone 2 validation: numerical diffusion of the three advection schemes.

The experiment is the classic one: a sharp-edged blob is carried around a rigidly rotating
velocity field for one full revolution.  Because the flow returns the fluid to where it
started, any difference from the initial condition is pure numerical error.  A *diffusive*
scheme smears the edge; a scheme without a limiter overshoots at the edge.  Both are measured.

Figures written to ``outputs/m2/``:

* ``rotation_methods.png`` -- initial blob and the result after one revolution for each
  scheme, colour-mapped on a common scale, plus an error map;
* ``diffusion_vs_resolution.png`` -- RMS error vs grid size for each scheme (log-log), with
  the fitted slopes;
* ``limiter_effect.png`` -- the same experiment on a top-hat (sharp-edged) blob with the
  limiter on and off: min/max reached, quantifying over/undershoot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mackerel.advect import AdvectionOptions, Scheme, advect_scalar
from mackerel.analytic import solid_body_rotation
from mackerel.diagnostics import scalar_integral
from mackerel.grid import Array, Grid, walls_bc

OUT = Path("outputs/m2")


def rotating_blob(
    n: int,
    nsteps: int,
    scheme: Scheme,
    limiter: bool = True,
    sharp: bool = False,
    radius: float = 0.07,
    distance: float = 0.25,
) -> tuple[Array, Array, Array, Array]:
    """Advect a blob one full revolution about the box centre.

    Returns ``(initial, final, u, v)``.  ``sharp`` switches from a Gaussian to a top-hat.
    """
    g = Grid(n, n, lx=1.0)
    X, Y = g.cell_coords()
    cx = 0.5 + distance
    if sharp:
        field = (((X - cx) ** 2 + (Y - 0.5) ** 2) < radius**2).astype(np.float64)
    else:
        field = np.exp(-((X - cx) ** 2 + (Y - 0.5) ** 2) / (2 * radius**2))
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    dt = 2.0 * np.pi / nsteps
    opts = AdvectionOptions(scheme=scheme, limiter=limiter)
    f = field
    for _ in range(nsteps):
        f = advect_scalar(f, u, v, g, bc, dt, opts)
    return field, f, u, v


def rms(a: Array, b: Array) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_resolution_study() -> dict[str, dict[str, list[float]]]:
    """RMS error after one revolution versus grid size, per scheme."""
    steps = {32: 120, 64: 240, 128: 480}
    out: dict[str, dict[str, list[float]]] = {}
    for scheme in Scheme:
        ns: list[float] = []
        errs: list[float] = []
        drift: list[float] = []
        for n, nsteps in steps.items():
            f0, f1, _, _ = rotating_blob(n, nsteps, scheme)
            ns.append(float(n))
            errs.append(rms(f0, f1))
            # mass over the *interior* of the domain, away from the walls
            drift.append(
                abs(scalar_integral(f1, Grid(n, n, lx=1.0)) - scalar_integral(f0, Grid(n, n, lx=1.0)))
                / scalar_integral(f0, Grid(n, n, lx=1.0))
            )
        out[scheme.value] = {"n": ns, "rms": errs, "mass_drift": drift}
    return out


def limiter_measurement() -> dict[str, dict[str, float]]:
    """Top-hat blob, limiter on vs off: extrema and RMS error after one revolution."""
    res: dict[str, dict[str, float]] = {}
    for scheme in (Scheme.MACCORMACK, Scheme.BFECC):
        for limiter in (True, False):
            f0, f1, _, _ = rotating_blob(80, 200, scheme, limiter=limiter, sharp=True, radius=0.14)
            res[f"{scheme.value}_{'limited' if limiter else 'unlimited'}"] = {
                "max": float(f1.max()),
                "min": float(f1.min()),
                "rms": rms(f0, f1),
            }
    return res


def _plots(study: dict[str, dict[str, list[float]]], lim: dict[str, dict[str, float]],
           outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # --- figure: initial vs each method after one revolution, plus error
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), layout="constrained")
    f0, f1, _, _ = rotating_blob(128, 480, Scheme.SEMI_LAGRANGIAN)
    axes[0, 0].imshow(f0, origin="lower", cmap="inferno", vmin=0, vmax=1)
    axes[0, 0].set_title("initial blob")
    axes[1, 0].axis("off")
    for k, scheme in enumerate(Scheme, start=1):
        _, f1, _, _ = rotating_blob(128, 480, scheme)
        axes[0, k].imshow(f1, origin="lower", cmap="inferno", vmin=0, vmax=1)
        axes[0, k].set_title(f"{scheme.value}\nRMS err {rms(f0, f1):.2e}")
        err = np.abs(f1 - f0)
        axes[1, k].imshow(err, origin="lower", cmap="magma", vmin=0, vmax=0.2)
        axes[1, k].set_title("|error|")
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0.0, 1.0), cmap=matplotlib.colormaps["inferno"]
    )
    fig.colorbar(sm, ax=axes[0, -1], label="dye", shrink=0.85)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("One full revolution of a Gaussian blob on 128^2 (480 steps)")
    fig.savefig(outdir / "rotation_methods.png", dpi=130)
    plt.close(fig)

    # --- figure: error vs resolution
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, data in study.items():
        n = np.array(data["n"])
        e = np.array(data["rms"])
        slope = float(np.polyfit(np.log(1.0 / n), np.log(e), 1)[0])
        ax.loglog(n, e, "o-", label=f"{name} (slope {slope:.2f})")
    ax.set_xlabel("grid n (n x n)")
    ax.set_ylabel("RMS error after one revolution")
    ax.set_title("Numerical diffusion of one full rotation")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "diffusion_vs_resolution.png", dpi=140)
    plt.close(fig)

    # --- figure: limiter effect (symlog: the unlimited schemes blow up by ~9 orders)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout="constrained")
    labels = list(lim)
    xpos = np.arange(len(labels), dtype=float)
    mx = np.array([lim[k]["max"] for k in labels])
    mn = np.array([lim[k]["min"] for k in labels])
    ax.bar(xpos - 0.2, mx, width=0.4, label="max value")
    ax.bar(xpos + 0.2, mn, width=0.4, label="min value")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="legal range of a 0/1 top-hat")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_yscale("symlog", linthresh=1e-3)
    for x, v in zip(xpos - 0.2, mx, strict=True):
        ax.text(x, v, f"{v:.2g}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    for x, v in zip(xpos + 0.2, mn, strict=True):
        ax.text(x, v, f"{v:.2g}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=8)
    ax.set_ylabel("value reached (symlog)")
    ax.set_title("Limiter on/off: a 0/1 top-hat advected one full revolution", pad=12)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "limiter_effect.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the milestone 2 experiments and print the measured numbers."""
    outdir = Path(argv[0]) if argv else OUT
    study = run_resolution_study()
    lim = limiter_measurement()
    _plots(study, lim, outdir)

    print("one full rotation, RMS error vs grid")
    for name, data in study.items():
        pairs = ", ".join(f"{int(n)}^2:{e:.3e}" for n, e in zip(data["n"], data["rms"], strict=True))
        slope = float(np.polyfit(np.log(1.0 / np.array(data["n"])), np.log(data["rms"]), 1)[0])
        print(f"  {name:16s} {pairs}   slope {slope:+.2f}")
        print(f"  {'':16s} mass drift " +
              " ".join(f"{d:.2e}" for d in data["mass_drift"]))

    print("\ntop-hat blob (0/1), one revolution, 80^2, 200 steps")
    for k, v in lim.items():
        print(f"  {k:24s} max {v['max']:+.4f}  min {v['min']:+.4f}  RMS {v['rms']:.3e}")

    print(f"\nfigures written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
