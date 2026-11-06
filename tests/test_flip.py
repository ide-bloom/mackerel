"""M7 PIC/FLIP and free-surface projection regressions."""

from __future__ import annotations

import numpy as np

from mackerel.flip import (
    FreeSurfaceWater,
    WaterConfig,
    _gather,
    _scatter,
    dam_break_particles,
)
from mackerel.grid import Grid, apply_velocity_bc, walls_bc
from mackerel.project import PoissonSystem, project_velocity


def test_wall_interpolation_has_nonnegative_partition_of_unity() -> None:
    grid = Grid(16, 12)
    x = np.array([1e-5, 1.0 - 1e-5])
    y = np.array([1e-5, grid.ly - 1e-5])
    values = np.array([2.0, -3.0])
    for component, shape in (("u", grid.shape_u), ("v", grid.shape_v)):
        total, weight = _scatter(values, x, y, grid.h, shape, component)
        assert np.all(weight >= 0.0)
        assert np.isclose(np.sum(weight), 2.0)
        # A constant field is reproduced exactly, including the wall-adjacent faces.
        ones, _ = _scatter(np.ones(2), x, y, grid.h, shape, component)
        normalised = np.divide(total, weight, out=np.zeros_like(total), where=weight > 1e-12)
        assert np.all(np.isfinite(normalised))
        assert np.isclose(np.sum(ones), 2.0)
        gathered = _gather(np.ones(shape), x, y, grid.h, component)
        assert np.allclose(gathered, 1.0)


def test_free_surface_air_is_dirichlet_not_solid_neumann() -> None:
    grid = Grid(16, 16)
    bc = walls_bc()
    rng = np.random.default_rng(42)
    u = rng.normal(size=grid.shape_u)
    v = rng.normal(size=grid.shape_v)
    apply_velocity_bc(u, v, bc)
    air = np.zeros(grid.shape_cell, dtype=bool)
    air[8:, :] = True
    result = project_velocity(
        u, v, grid, bc, 0.01, solid=air, free_surface=True, tol=1e-10
    )
    assert np.all(result.p[air] == 0.0)
    assert result.div_max < 1e-7
    system = PoissonSystem.from_bc(16, 16, grid.h, bc, solid=air, free_surface=True)
    # The top liquid row has an atmospheric diagonal entry in addition to its liquid
    # neighbour couplings.  A solid-Neumann operator would not have this entry.
    obstacle = PoissonSystem.from_bc(16, 16, grid.h, bc, solid=air, free_surface=False)
    assert np.all(system.diag[7, :] > obstacle.diag[7, :])
    other = project_velocity(u, v, grid, bc, 0.01, solid=air, free_surface=False)
    assert not np.allclose(result.p, other.p)


def test_particle_seeding_and_closed_volume() -> None:
    grid = Grid(16, 12)
    particles = dam_break_particles(grid, width=0.25, height=0.5)
    assert len(particles) == 4 * 4 * 8
    water = FreeSurfaceWater(grid, particles, config=WaterConfig(dt=0.0005))
    initial_volume = water.water_volume()
    water.run_steps(8)
    assert len(water.particles) == 4 * 4 * 8
    assert water.water_volume() == initial_volume
    assert np.all(water.particles.x > 0.0)
    assert np.all(water.particles.x < grid.lx)
    assert np.all(water.particles.y > 0.0)
    assert np.all(water.particles.y < grid.ly)


def test_short_dam_break_pic_and_flip_are_finite_and_distinct() -> None:
    results: list[FreeSurfaceWater] = []
    for blend in (1.0, 0.05):
        grid = Grid(20, 16)
        particles = dam_break_particles(grid, width=0.25, height=0.5)
        water = FreeSurfaceWater(
            grid, particles,
            config=WaterConfig(dt=0.0005, pic_blend=blend, pressure_tol=1e-8),
        )
        water.run_steps(30)
        assert np.all(np.isfinite(water.particles.x))
        assert np.all(np.isfinite(water.particles.u))
        assert water.front_position() > 0.23
        assert water.history[-1].divergence < 1e-6
        results.append(water)
    assert abs(results[0].front_position() - results[1].front_position()) > 1e-5


def test_run_until_clamps_last_step() -> None:
    grid = Grid(12, 10)
    water = FreeSurfaceWater(
        grid, dam_break_particles(grid), config=WaterConfig(dt=0.002)
    )
    water.run_until(0.003)
    assert np.isclose(water.time, 0.003)
    assert np.isclose(water.history[-1].dt, 0.001)


def test_numba_transfer_matches_numpy_when_available() -> None:
    import pytest

    from mackerel.flip_numba import numba_available, scatter_numba

    if not numba_available():
        pytest.skip("optional Numba dependency is not installed")
    grid = Grid(16, 12)
    particles = dam_break_particles(grid)
    for component, shape in (("u", grid.shape_u), ("v", grid.shape_v)):
        expected = _scatter(particles.u, particles.x, particles.y, grid.h, shape, component)
        actual = scatter_numba(particles.u, particles.x, particles.y, grid.h, shape, component)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-13, atol=1e-13)


def test_numba_water_backend_matches_numpy_when_available() -> None:
    import pytest

    from mackerel.flip_numba import numba_available

    if not numba_available():
        pytest.skip("optional Numba dependency is not installed")
    grid = Grid(12, 10)
    reference = FreeSurfaceWater(
        grid, dam_break_particles(grid),
        config=WaterConfig(dt=0.0005, transfer_backend="numpy"),
    )
    accelerated = FreeSurfaceWater(
        grid, dam_break_particles(grid),
        config=WaterConfig(dt=0.0005, transfer_backend="numba"),
    )
    reference.run_steps(6)
    accelerated.run_steps(6)
    np.testing.assert_allclose(accelerated.particles.x, reference.particles.x, atol=1e-12)
    np.testing.assert_allclose(accelerated.particles.y, reference.particles.y, atol=1e-12)
    np.testing.assert_allclose(accelerated.particles.u, reference.particles.u, atol=1e-12)
    np.testing.assert_allclose(accelerated.particles.v, reference.particles.v, atol=1e-12)
