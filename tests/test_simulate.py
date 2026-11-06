"""Integration tests for the split 2D Navier--Stokes solver."""

from __future__ import annotations

import numpy as np
import pytest

from mackerel.advect import Scheme
from mackerel.diagnostics import divergence_field
from mackerel.grid import (
    BoundaryConditions,
    Grid,
    Kind,
    SideBC,
    periodic_bc,
    walls_bc,
)
from mackerel.simulate import (
    Simulation,
    SimulationConfig,
    buoyancy_force,
    run,
)


def channel_bc() -> BoundaryConditions:
    p = SideBC(Kind.PERIODIC)
    w = SideBC(Kind.NO_SLIP)
    return BoundaryConditions(p, p, w, w)


def test_taylor_green_factory_keeps_viscosity() -> None:
    """The convenience constructor must put ``nu`` in the actual time-step config."""
    sim = Simulation.taylor_green(Grid(16, 16), nu=0.01)
    assert sim.config.nu == 0.01
    assert sim.bc.is_x_periodic and sim.bc.is_y_periodic


def test_string_scheme_is_coerced_to_enum() -> None:
    sim = Simulation.taylor_green(Grid(8, 8), scheme="maccormack")
    assert sim.config.scheme is Scheme.MACCORMACK
    with pytest.raises(ValueError):
        SimulationConfig(scheme="not-a-scheme")  # type: ignore[arg-type]


def test_taylor_green_step_is_divergence_free_and_decays() -> None:
    """Advection + implicit viscosity + projection preserves the TG decay trend."""
    grid = Grid(32, 32, lx=1.0)
    nu = 0.01
    sim = Simulation.taylor_green(
        grid, nu=nu, dt=0.0025, scheme=Scheme.MACCORMACK, pressure_tol=1e-9
    )
    initial = sim.energy()
    infos = sim.run_until(0.05)
    exact = initial * np.exp(-4.0 * (2.0 * np.pi) ** 2 * nu * sim.time)
    # Measured at 32^2, dt=0.0025: relative error -1.48%; the remaining error is
    # semi-Lagrangian/MacCormack numerical diffusion plus the first-order operator split.
    assert sim.energy() == pytest.approx(exact, rel=0.03)
    assert sim.energy() < initial
    assert infos[-1].div_max < 1e-7
    assert np.max(np.abs(divergence_field(sim.u, sim.v, grid))) < 1e-7
    assert len(infos) == 20
    assert sim.time == pytest.approx(0.05)


def test_projection_does_not_change_an_already_divergence_free_state_much() -> None:
    grid = Grid(24, 24)
    sim = Simulation.taylor_green(grid, nu=0.0, dt=0.001)
    before = sim.energy()
    sim.step()
    # One advection step is not exactly energy conserving, but the solve must remain finite
    # and divergence free; this catches accidental pressure-sign/regression errors.
    assert np.isfinite(sim.energy())
    assert sim.energy() > 0.9 * before
    assert sim.history[-1].div_max < 1e-7


def test_fixed_dt_and_cfl_dt() -> None:
    grid = Grid(16, 16)
    fixed = Simulation.taylor_green(grid, dt=0.002)
    assert fixed.dt_for_cfl() == pytest.approx(0.002)
    automatic = Simulation.taylor_green(grid, cfl=0.5)
    expected = min(grid.h, 0.5 * grid.h / automatic.max_speed())
    assert automatic.dt_for_cfl() == pytest.approx(expected)
    info = automatic.step()
    assert info.dt == pytest.approx(expected)
    assert info.cfl <= 0.5 + 1e-12


def test_resting_flow_has_a_bounded_first_step() -> None:
    sim = Simulation.at_rest(Grid(8, 8), walls_bc(), cfl=1.0)
    assert sim.dt_for_cfl() == pytest.approx(sim.grid.h)
    info = sim.step()
    assert info.dt == pytest.approx(sim.grid.h)
    assert np.all(sim.u == 0.0) and np.all(sim.v == 0.0)


def test_run_until_does_not_overshoot() -> None:
    sim = Simulation.at_rest(Grid(8, 8), walls_bc(), dt=0.03)
    sim.run_until(0.1)
    assert sim.time == pytest.approx(0.1)


def test_body_force_maps_cell_values_to_mac_faces() -> None:
    grid = Grid(4, 4)
    sim = Simulation.at_rest(grid, walls_bc())
    fx = np.arange(16, dtype=float).reshape(4, 4)
    fy = 2.0 * fx
    sim.apply_force(fx, fy, dt=0.1)
    np.testing.assert_allclose(sim.u[:, 1:-1], 0.1 * 0.5 * (fx[:, :-1] + fx[:, 1:]))
    np.testing.assert_allclose(sim.v[1:-1, :], 0.1 * 0.5 * (fy[:-1, :] + fy[1:, :]))
    # prescribed normal faces remain exactly at their wall values
    assert np.all(sim.u[:, [0, -1]] == 0.0)
    assert np.all(sim.v[[0, -1], :] == 0.0)


def test_buoyancy_sign_and_mean_removal() -> None:
    grid = Grid(4, 4)
    temperature = np.zeros(grid.shape_cell)
    temperature[2:, :] = 1.0
    force = buoyancy_force(grid, temperature, gravity=9.81, beta=1.0)
    # hot fluid rises (positive y), cold fluid sinks; the spatial mean force is zero
    assert np.all(force[2:, :] > 0.0)
    assert np.all(force[:2, :] < 0.0)
    assert float(np.mean(force)) == pytest.approx(0.0, abs=1e-14)


def test_scalar_diffusivity_is_independent_of_velocity_viscosity() -> None:
    grid = Grid(16, 16)
    sim = Simulation.at_rest(
        grid, periodic_bc(), dt=0.01, nu=0.0, scalar_diffusivity=0.1
    )
    X, Y = grid.cell_coords()
    field = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    sim.add_scalar("temperature", field)
    sim.step()
    assert np.max(np.abs(sim.scalars["temperature"] - field)) > 1e-3
    assert np.max(np.abs(sim.u)) == 0.0
    assert np.max(np.abs(sim.v)) == 0.0


def test_channel_body_force_produces_poiseuille_profile() -> None:
    """A small channel run reaches the analytical parabolic steady state to second order."""
    grid = Grid(16, 16, lx=1.0)
    force, nu = 0.1, 0.05
    sim = Simulation.at_rest(
        grid, channel_bc(), nu=nu, dt=0.01, pressure_tol=1e-10, buoyancy=(force, 0.0)
    )
    # The viscous time scale is H^2/nu = 20; this is long enough for the coarse-grid
    # steady state and keeps the test inexpensive.
    sim.run_until(30.0)
    uc = grid.u_to_centers(sim.u)
    y = grid.y_centers()
    exact = (force / (2.0 * nu) * y * (grid.ly - y))[:, None]
    error = float(np.sqrt(np.mean((uc - exact) ** 2)))
    # Measured L2 error at 16^2 after steady state: 9.77e-4 (relative 3.9e-3).
    assert error < 2e-3
    assert np.max(np.abs(sim.v)) < 1e-10


def test_run_helper_returns_simulation() -> None:
    grid = Grid(8, 8)
    sim = run(grid, periodic_bc(), SimulationConfig(dt=0.001), 0.003)
    assert sim.time == pytest.approx(0.003)
    assert sim.u.shape == grid.shape_u
