"""Milestone 5 cavity reference and smoke tests (the long benchmark is in validation/)."""

from __future__ import annotations

import numpy as np
from validation.lid_cavity import centreline_profiles, reference

from mackerel.grid import Grid, walls_bc
from mackerel.simulate import Simulation


def test_ghia_tables_have_complete_re100_re400_re1000_data() -> None:
    for re in (100, 400, 1000):
        ref = reference(re)
        assert set(ref) == {"u_vertical_centerline", "v_horizontal_centerline"}
        for coordinates, values in ref.values():
            assert coordinates.shape == (17,)
            assert values.shape == (17,)
            assert np.all(np.diff(coordinates) > 0.0)
            assert np.isfinite(values).all()
    # The disputed Re=400 datum is preserved, not silently corrected.
    x, v = reference(400)["v_horizontal_centerline"]
    assert v[np.isclose(x, 0.9063)][0] == -0.23827


def test_centreline_profiles_include_physical_wall_values() -> None:
    grid = Grid(8, 8)
    sim = Simulation.at_rest(grid, walls_bc((1.0, 0.0)), dt=0.01)
    y, u, x, v = centreline_profiles(sim)
    assert y[0] == 0.0 and y[-1] == 1.0
    assert u[0] == 0.0 and u[-1] == 1.0
    assert x[0] == 0.0 and x[-1] == 1.0
    assert v[0] == 0.0 and v[-1] == 0.0


def test_cavity_short_run_starts_a_divergence_free_vortex() -> None:
    grid = Grid(16, 16)
    sim = Simulation.at_rest(
        grid, walls_bc((1.0, 0.0)), nu=0.01, dt=0.02, pressure_tol=1e-8
    )
    sim.run_until(0.2)
    assert sim.energy() > 0.0
    assert np.isfinite(sim.u).all() and np.isfinite(sim.v).all()
    assert sim.history[-1].div_max < 1e-6
    # normal wall velocities remain exactly zero; the tangential lid is imposed by ghosts
    assert np.all(sim.u[:, [0, -1]] == 0.0)
    assert np.all(sim.v[[0, -1], :] == 0.0)
