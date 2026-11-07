"""Regressions for the dam-break front-diagnosis driver."""

from __future__ import annotations

import numpy as np
import pytest
from validation.dam_break_diagnosis import FRONT_KEYS, front_metrics, run_case, score

from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig
from mackerel.grid import Grid


def _column_water(n_particles_per_cell: int = 4) -> FreeSurfaceWater:
    grid = Grid(12, 9)
    particles = ParticleSet.from_region(
        grid, lambda X, Y: (X <= 0.25) & (Y <= 0.5),
        particles_per_cell=n_particles_per_cell, seed=0,
    )
    return FreeSurfaceWater(
        grid, particles, particles_per_cell=n_particles_per_cell,
        config=WaterConfig(dt=0.001),
    )


def test_front_estimators_are_ordered_and_inside_the_domain() -> None:
    water = _column_water()
    metrics = front_metrics(water)
    assert set(metrics) == set(FRONT_KEYS)
    for value in metrics.values():
        assert 0.0 < value <= water.grid.lx + 1e-12
    # The half-cell offset shifts the rightmost estimator by exactly h/2.
    assert np.isclose(metrics["rightmost_half"] - metrics["rightmost"], 0.5 * water.grid.h)
    # Percentiles are particle positions, so they cannot exceed the raw maximum.
    for key in ("p99", "p995"):
        assert metrics[key] <= metrics["rightmost"] + 1e-12
    # Cell-column estimators report the right edge of the last qualifying column, so they sit
    # up to one cell above the rightmost particle -- a convention, not an inconsistency.
    for key in ("column_k4", "column_k8", "thickness_2h"):
        assert metrics[key] <= metrics["rightmost"] + water.grid.h + 1e-12


def test_higher_seeding_density_is_not_rejected_when_reseeding_is_off() -> None:
    """A 9-particle-per-cell column must construct: the reseed caps only apply when enabled."""
    water = _column_water(9)
    # 3 columns and 6 rows of cell centres satisfy the region, at 9 particles each.
    assert len(water.particles) == 9 * 3 * 6
    # Nominal volume uses the seeded density, so it matches the geometric column volume.
    assert np.isclose(water.water_volume(), 0.25 * 0.5, rtol=0.25)
    # With reseeding enabled the same density is rejected: the donor cap must be raised first.
    with pytest.raises(ValueError):
        FreeSurfaceWater(
            Grid(12, 9), ParticleSet(np.array([0.1]), np.array([0.1]),
                                     np.zeros(1), np.zeros(1)),
            particles_per_cell=9,
            config=WaterConfig(reseed=True, max_particles_per_cell=8),
        )
    # ... and it constructs once the cap admits the density.
    enabled = FreeSurfaceWater(
        Grid(12, 9), ParticleSet(np.array([0.1]), np.array([0.1]), np.zeros(1), np.zeros(1)),
        particles_per_cell=9, config=WaterConfig(reseed=True, max_particles_per_cell=16),
    )
    assert enabled.reseed_target() == 9


def test_short_case_scores_every_convention_against_the_reference() -> None:
    result = run_case(48, end_tau=1.0)
    scores = score(result)
    assert set(scores) == set(FRONT_KEYS)
    for value in scores.values():
        assert np.isfinite(value) and value >= 0.0
    assert result["volume_relative_error"] == 0.0
    assert len(result["tau"]) == len(result["fronts"]["rightmost"]) == result["steps"]


def test_the_milestone_regression_is_reproduced_by_the_driver() -> None:
    """The reported 48 -> 96 front-RMSE rise must still be visible in this driver's output."""
    coarse = score(run_case(48, end_tau=1.6))
    fine = score(run_case(96, end_tau=1.6))
    assert fine["rightmost_half"] > coarse["rightmost_half"]
