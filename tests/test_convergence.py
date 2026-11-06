"""Small regressions for the Taylor--Green convergence driver."""

from __future__ import annotations

import numpy as np
from validation.convergence import observed_order, run_case, strang_step

from mackerel.advect import Scheme
from mackerel.grid import Grid
from mackerel.simulate import Simulation


def test_observed_order_recovers_a_known_second_order_sequence() -> None:
    result = observed_order(
        np.array([1.0, 0.5, 0.25]), np.array([0.16, 0.04, 0.01])
    )
    assert np.isclose(result["order"], 2.0)
    assert np.allclose(result["pairwise"], 2.0)


def test_strang_control_is_finite_and_divergence_free() -> None:
    sim = Simulation.taylor_green(
        Grid(16, 16), nu=0.01, dt=0.0025, scheme=Scheme.MACCORMACK
    )
    div = strang_step(sim, 0.0025)
    assert np.isfinite(sim.energy())
    assert div < 1e-7
    assert np.isclose(sim.time, 0.0025)


def test_short_convergence_case_has_consistent_error_metrics() -> None:
    row = run_case(16, 0.0025, 0.01)
    assert row["velocity_l2"] >= 0.0
    assert row["sampled_velocity_l2"] >= 0.0
    assert row["max_divergence"] < 1e-7
    assert np.isclose(row["time"], 0.01)
