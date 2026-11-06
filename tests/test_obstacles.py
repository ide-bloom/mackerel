"""Tests for immersed solid masks and moving-obstacle boundary conditions."""

from __future__ import annotations

import numpy as np

from mackerel.grid import Grid, periodic_bc, walls_bc
from mackerel.obstacles import (
    MovingObstacle,
    apply_solid_velocity,
    box_mask,
    circle_mask,
    solid_face_masks,
)
from mackerel.project import project_velocity
from mackerel.simulate import Simulation, SimulationConfig


def test_circle_and_box_masks_have_expected_shapes() -> None:
    grid = Grid(32, 16, lx=2.0)
    circle = circle_mask(grid, (0.5, 0.25), 0.1)
    box = box_mask(grid, (0.2, 0.1), (0.4, 0.3))
    assert circle.shape == grid.shape_cell and circle.dtype == np.bool_
    assert box.shape == grid.shape_cell
    assert circle.any() and box.any()
    um, vm = solid_face_masks(circle)
    assert um.shape == grid.shape_u and vm.shape == grid.shape_v
    assert um.any() and vm.any()


def test_static_solid_faces_survive_projection() -> None:
    grid = Grid(24, 24)
    solid = circle_mask(grid, (0.5, 0.5), 0.15)
    rng = np.random.default_rng(4)
    u = rng.standard_normal(grid.shape_u)
    v = rng.standard_normal(grid.shape_v)
    apply_solid_velocity(u, v, solid)
    um, vm = solid_face_masks(solid)
    result = project_velocity(u, v, grid, walls_bc(), 0.1, solid=solid, tol=1e-9)
    assert np.all(result.u[um] == 0.0)
    assert np.all(result.v[vm] == 0.0)
    assert np.isfinite(result.u).all() and np.isfinite(result.v).all()


def test_moving_solid_velocity_is_restored_after_a_step() -> None:
    grid = Grid(24, 24)
    solid = circle_mask(grid, (0.5, 0.5), 0.15)
    sim = Simulation(grid, periodic_bc(), SimulationConfig(dt=0.001), solid=solid, solid_velocity=(0.2, -0.1))
    um, vm = solid_face_masks(solid)
    sim.step()
    np.testing.assert_allclose(sim.u[um], 0.2)
    np.testing.assert_allclose(sim.v[vm], -0.1)


def test_set_solid_invalidates_system_and_updates_velocity() -> None:
    grid = Grid(16, 16)
    sim = Simulation(grid, walls_bc(), SimulationConfig(dt=0.001))
    sim._system_for()
    mask = box_mask(grid, (0.4, 0.4), (0.6, 0.6))
    sim.set_solid(mask, (0.1, 0.0))
    assert sim._system is None
    um, _ = solid_face_masks(mask)
    assert np.all(sim.u[um] == 0.1)
    sim.step()
    assert np.all(sim.u[um] == 0.1)


def test_moving_obstacle_translates_and_applies_its_velocity() -> None:
    grid = Grid(32, 32)
    obstacle = MovingObstacle("circle", (0.3, 0.5), 0.08, velocity=(0.1, 0.0))
    u = np.zeros(grid.shape_u)
    v = np.zeros(grid.shape_v)
    mask0 = obstacle.apply(u, v, grid, 0.0)
    mask1 = obstacle.apply(u, v, grid, 1.0)
    assert not np.array_equal(mask0, mask1)
    um, _ = solid_face_masks(mask1)
    assert np.all(u[um] == 0.1)


def test_outflow_is_not_a_zero_velocity_dirichlet_wall() -> None:
    from mackerel.diffuse import diffuse
    from mackerel.grid import BoundaryConditions, Kind, SideBC, apply_velocity_bc, divergence

    grid = Grid(16, 16)
    bc = BoundaryConditions(
        SideBC(Kind.INFLOW, velocity=(1.0, 0.0)), SideBC(Kind.OUTFLOW),
        SideBC(Kind.NO_SLIP), SideBC(Kind.NO_SLIP),
    )
    u = np.ones(grid.shape_u)
    v = np.zeros(grid.shape_v)
    apply_velocity_bc(u, v, bc)
    result = diffuse(u, v, grid, bc, 0.001, 0.003)
    assert np.min(result.u[:, -1]) > 0.9
    projected = project_velocity(result.u, result.v, grid, bc, 0.003, tol=1e-8)
    assert projected.div_max < 1e-6
    assert np.max(np.abs(divergence(projected.u, projected.v, grid.h))) < 1e-6
