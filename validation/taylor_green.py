"""Milestone 4 Taylor--Green validation.

The periodic Taylor--Green vortex has the closed-form kinetic energy
``E(t) = E(0) exp(-4 k^2 nu t)``.  This script runs the *complete* split solver (advection,
viscosity and pressure projection), not just the diffusion routine, and records the measured
energy error, decay rate and divergence.  A fixed ``dt`` refinement sweep isolates spatial
error from the usual CFL-driven ``dt ~ h`` coupling.

Run ``python -m validation.taylor_green``; figures and CSV/JSON data go to ``outputs/m4``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme
from mackerel.grid import Grid
from mackerel.simulate import Simulation, StepInfo

OUT = Path("outputs/m4")
K = 2.0 * np.pi


def exact_energy(e0: float, t: float | np.ndarray, nu: float) -> float | np.ndarray:
    return e0 * np.exp(-4.0 * K**2 * nu * t)


def one_run(
    n: int,
    scheme: Scheme,
    nu: float = 0.01,
    dt: float = 0.0025,
    end_time: float = 0.2,
) -> dict[str, Any]:
    grid = Grid(n, n, lx=1.0)
    sim = Simulation.taylor_green(
        grid, nu=nu, dt=dt, scheme=scheme, pressure_tol=1e-9, pressure_maxiter=500
    )
    e0 = sim.energy()
    times = [0.0]
    energies = [e0]
    def record(state: Simulation, _info: StepInfo) -> None:
        times.append(state.time)
        energies.append(state.energy())

    infos = sim.run_until(end_time, record)
    t = np.asarray(times)
    e = np.asarray(energies)
    exact = np.asarray(exact_energy(e0, t, nu))
    # Fit only the positive part of the decay curve; this is a measured effective rate,
    # including numerical diffusion, rather than a claim that the analytic rate was imposed.
    positive = e > 0.0
    rate = float(-np.polyfit(t[positive], np.log(e[positive]), 1)[0])
    exact_rate = float(4.0 * K**2 * nu)
    return {
        "n": n,
        "scheme": scheme.value,
        "nu": nu,
        "dt": dt,
        "end_time": float(sim.time),
        "steps": len(infos),
        "e0": e0,
        "energy": float(e[-1]),
        "exact_energy": float(exact[-1]),
        "relative_error": float(e[-1] / exact[-1] - 1.0),
        "max_absolute_energy_error": float(np.max(np.abs(e - exact))),
        "measured_decay_rate": rate,
        "exact_decay_rate": exact_rate,
        "max_divergence": float(max(i.div_max for i in infos)),
        "times": t,
        "energies": e,
        "exact_energies": exact,
    }


def main(argv: list[str] | None = None) -> int:
    outdir = Path(argv[0]) if argv else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    nu, dt, end_time = 0.01, 0.0025, 0.2

    # Scheme comparison at a fixed grid and time step.
    comparison = [one_run(64, scheme, nu, dt, end_time) for scheme in Scheme]
    # Fixed-dt refinement: changing n does not silently change the number of time steps.
    refinement = [one_run(n, Scheme.MACCORMACK, nu, dt, end_time) for n in (32, 64, 128)]

    serial: list[dict[str, Any]] = []
    for row in comparison + refinement:
        serial.append({k: v for k, v in row.items() if not isinstance(v, np.ndarray)})
    (outdir / "taylor_green.json").write_text(json.dumps(serial, indent=2) + "\n")
    with (outdir / "taylor_green.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    for row in comparison:
        ax.plot(row["times"], row["energies"], label=f"{row['scheme']} (numerical)")
    ref = comparison[0]
    ax.plot(ref["times"], ref["exact_energies"], "k--", lw=2, label="analytic")
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title(f"Taylor-Green decay, 64², nu={nu}, dt={dt}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "energy_decay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    ns = np.array([r["n"] for r in refinement], dtype=float)
    errors = np.array([abs(float(r["relative_error"])) for r in refinement])
    slope = float(np.polyfit(np.log(1.0 / ns), np.log(errors), 1)[0])
    ax.loglog(ns, errors, "o-", label=f"measured (slope {slope:.2f})")
    ax.loglog(ns, errors[0] * (ns[0] / ns) ** 2, "k:", label="second-order reference")
    ax.set_xlabel("grid n")
    ax.set_ylabel("relative energy error at t=0.2")
    ax.set_title("Fixed-time-step Taylor-Green refinement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "refinement.png", dpi=150)
    plt.close(fig)

    print("Taylor-Green full-solver validation")
    for row in comparison:
        print(
            f"  {row['scheme']:16s} E={row['energy']:.8f} "
            f"exact={row['exact_energy']:.8f} rel={row['relative_error']:+.3e} "
            f"rate={row['measured_decay_rate']:.5f} "
            f"div={row['max_divergence']:.2e}"
        )
    print("fixed-dt refinement (MacCormack)")
    for row in refinement:
        print(
            f"  {row['n']:4d}² rel={row['relative_error']:+.3e} "
            f"max|dE|={row['max_absolute_energy_error']:.3e}"
        )
    print(f"refinement slope (absolute relative energy error): {slope:.2f}")
    print(f"figures and data written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
