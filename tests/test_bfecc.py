"""Regression for the BFECC staggered-face sampling bug."""

from __future__ import annotations

import numpy as np

from mackerel.advect import Scheme
from mackerel.grid import Grid
from mackerel.simulate import Simulation


def test_bfecc_full_taylor_green_has_no_v_face_offset_energy_loss() -> None:
    """The corrected BFECC v sweep stays within 2% of analytic energy at t=.2."""
    grid = Grid(64, 64)
    sim = Simulation.taylor_green(
        grid, nu=0.01, dt=0.0025, scheme=Scheme.BFECC,
        pressure_tol=1e-9, pressure_maxiter=500,
    )
    e0 = sim.energy()
    sim.run_until(0.2)
    exact = e0 * np.exp(-4.0 * (2.0 * np.pi) ** 2 * 0.01 * 0.2)
    # The legacy u-offset v sweep gave approximately -22.9% here; this bound catches it.
    assert abs(sim.energy() / exact - 1.0) < 0.02
