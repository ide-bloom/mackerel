"""Fast numerical-regression subset used by CI.

The full test suite covers correctness; this script covers *accuracy* on two small problems so
that a numerical regression (a changed stencil offset, a broken projection, a lost limiter)
fails CI instead of only changing a number in a milestone report.

Both thresholds were measured on this repository's NumPy reference path and are frozen here
with an explicit margin, so the check flags a real regression rather than ordinary
floating-point noise.  Nothing here tunes a solver parameter: the configurations are the
documented M4/M7 settings at a smaller grid, and the numbers below are outputs.

Run ``python -m validation.ci_subset``.  Exit status is non-zero if any check fails, and the
JSON block on stdout is the machine-readable form of the same result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.advect import Scheme
from mackerel.diffuse import diffuse
from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig
from mackerel.grid import Grid, apply_velocity_bc, walls_bc
from mackerel.project import project_velocity
from mackerel.simulate import Simulation

K = 2.0 * np.pi
OUT = Path("outputs/ci")


def taylor_green_split_step(n: int = 32, nu: float = 0.01, dt: float = 0.0025,
                            end_time: float = 0.2) -> dict[str, float]:
    """Complete split solver (advection + viscosity + projection) against the exact decay.

    Measured on this repository: ``relative_energy_error = -1.300313e-02`` and
    ``max_divergence = 7.189e-11`` at 32x32.  The tolerance below is ``|error| < 0.02``, i.e.
    about 1.5x the measured magnitude, which leaves room for platform-level floating-point
    differences while failing on the legacy BFECC offset bug (measured ``-2.294e-01``).
    """
    sim = Simulation.taylor_green(
        Grid(n, n), nu=nu, dt=dt, scheme=Scheme.BFECC,
        pressure_tol=1e-9, pressure_maxiter=500,
    )
    e0 = sim.energy()
    sim.run_until(end_time)
    exact = e0 * np.exp(-4.0 * K**2 * nu * end_time)
    return {
        "n": float(n),
        "relative_energy_error": float(sim.energy() / exact - 1.0),
        "max_divergence": float(max(info.div_max for info in sim.history)),
    }


def bfecc_offset_control(n: int = 32, nu: float = 0.01, dt: float = 0.0025) -> dict[str, float]:
    """One BFECC sweep + viscosity + projection, corrected versus legacy ``v`` offsets.

    Measured on this repository: corrected ``energy_ratio = 0.984116``, legacy
    ``0.962012``.  The check asserts both an absolute band on the corrected step and that the
    legacy control stays measurably worse, which is the actual property that was fixed.
    """
    from validation.bfecc import bfecc_variant

    grid = Grid(n, n)
    bc = walls_bc()
    sim = Simulation.taylor_green(grid, nu=nu, dt=dt, scheme=Scheme.MACCORMACK, pressure_tol=1e-9)
    u0, v0 = sim.u.copy(), sim.v.copy()
    apply_velocity_bc(u0, v0, bc)
    energy0 = 0.5 * (
        float(np.sum(grid.u_to_centers(u0) ** 2)) + float(np.sum(grid.v_to_centers(v0) ** 2))
    ) * grid.cell_area
    out: dict[str, float] = {"n": float(n), "energy_initial": energy0}
    for label, legacy in (("corrected", False), ("legacy", True)):
        u, v = bfecc_variant(u0, v0, grid, bc, dt, legacy_v_offsets=legacy)
        result = diffuse(u, v, grid, bc, nu, dt)
        projected = project_velocity(result.u, result.v, grid, bc, dt, tol=1e-9, maxiter=500)
        energy = 0.5 * (
            float(np.sum(grid.u_to_centers(projected.u) ** 2))
            + float(np.sum(grid.v_to_centers(projected.v) ** 2))
        ) * grid.cell_area
        out[f"energy_ratio_{label}"] = energy / energy0
        out[f"max_divergence_{label}"] = float(projected.div_max)
    return out


def water_conservation(n: int = 24) -> dict[str, float]:
    """A short PIC/FLIP run: particle count and nominal volume must be exactly conserved.

    Measured on this repository: 288 particles in, 288 out, volume relative change exactly
    ``0.0``, ``max_divergence < 1e-6``.  The check is exact equality, not a tolerance, because
    a closed wall must not lose water.
    """
    grid = Grid(n, 3 * n // 4)
    particles = ParticleSet.from_region(
        grid, lambda X, Y: (X <= 0.25) & (Y <= 0.5), particles_per_cell=4, seed=0
    )
    water = FreeSurfaceWater(
        grid, particles,
        config=WaterConfig(dt=0.0005, pic_blend=0.05, pressure_tol=1e-8, pressure_maxiter=1200),
    )
    count = len(water.particles)
    volume = water.water_volume()
    water.run_steps(40)
    return {
        "n": float(n),
        "particles_initial": float(count),
        "particles_final": float(len(water.particles)),
        "volume_initial": volume,
        "volume_final": water.water_volume(),
        "volume_relative_error": abs(water.water_volume() - volume) / volume,
        "max_divergence": float(max(info.divergence for info in water.history)),
    }


def main(_argv: list[str] | None = None) -> int:
    checks: dict[str, Any] = {}
    failures: list[str] = []

    tg = taylor_green_split_step()
    checks["taylor_green_split_step"] = tg
    # Margin: measured 1.30e-2, bound 2.0e-2 (1.5x), legacy bug 2.29e-1.
    if abs(tg["relative_energy_error"]) >= 0.02:
        failures.append(
            f"taylor_green energy error {tg['relative_energy_error']:+.3e} exceeds 2.0e-2"
        )
    if tg["max_divergence"] >= 1e-8:
        failures.append(f"taylor_green max divergence {tg['max_divergence']:.3e} exceeds 1e-8")

    bf = bfecc_offset_control()
    checks["bfecc_offset_control"] = bf
    # Margin: measured 0.9841, band 0.975..0.995; legacy measured 0.9620.
    if not 0.975 < bf["energy_ratio_corrected"] < 0.995:
        failures.append(
            f"BFECC corrected energy ratio {bf['energy_ratio_corrected']:.6f} outside 0.975..0.995"
        )
    if bf["energy_ratio_legacy"] >= bf["energy_ratio_corrected"] - 0.005:
        failures.append(
            "the legacy v-offset control no longer shows its energy loss "
            f"({bf['energy_ratio_legacy']:.6f} vs {bf['energy_ratio_corrected']:.6f})"
        )

    water = water_conservation()
    checks["water_conservation"] = water
    if water["particles_final"] != water["particles_initial"]:
        failures.append(
            f"particle count changed {water['particles_initial']:.0f} -> "
            f"{water['particles_final']:.0f}"
        )
    if water["volume_relative_error"] == 0.0 and water["volume_final"] != water["volume_initial"]:
        failures.append("volume drifted despite an unchanged particle count")
    if water["volume_relative_error"] > 1e-12:
        failures.append(f"volume relative error {water['volume_relative_error']:.3e} exceeds 1e-12")
    if water["max_divergence"] >= 1e-6:
        failures.append(f"water max divergence {water['max_divergence']:.3e} exceeds 1e-6")

    report = {"status": "pass" if not failures else "fail", "failures": failures, "checks": checks}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ci_subset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
