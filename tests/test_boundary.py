"""Milestone 1: boundary conditions, one test family per kind."""

from __future__ import annotations

import numpy as np
import pytest

from mackerel.analytic import taylor_green
from mackerel.grid import (
    BoundaryConditions,
    Grid,
    Kind,
    Side,
    SideBC,
    apply_velocity_bc,
    bilinear,
    curl_streamfunction,
    pad_scalar,
    pad_velocity,
    periodic_bc,
    walls_bc,
)


def _unit() -> Grid:
    return Grid(8, 6, lx=1.0)


# --------------------------------------------------------------------------------------
# periodicity bookkeeping
# --------------------------------------------------------------------------------------
def test_periodic_must_come_in_pairs() -> None:
    bad = BoundaryConditions(SideBC(Kind.PERIODIC), SideBC(Kind.NO_SLIP),
                             SideBC(Kind.NO_SLIP), SideBC(Kind.NO_SLIP))
    with pytest.raises(ValueError):
        bad.validate()


def test_periodic_flags() -> None:
    assert periodic_bc().is_x_periodic and periodic_bc().is_y_periodic
    assert not walls_bc().is_x_periodic


# --------------------------------------------------------------------------------------
# no-slip walls, including the moving lid
# --------------------------------------------------------------------------------------
def test_no_slip_stationary_walls() -> None:
    g = _unit()
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, g.shape_u)
    v = rng.uniform(-1, 1, g.shape_v)
    apply_velocity_bc(u, v, walls_bc())
    np.testing.assert_allclose(u[:, 0], 0.0)
    np.testing.assert_allclose(u[:, -1], 0.0)
    np.testing.assert_allclose(v[0, :], 0.0)
    np.testing.assert_allclose(v[-1, :], 0.0)


def test_moving_lid_normal_velocity_is_zero_but_tangential_is_one() -> None:
    """The lid slides in x.  Its *normal* (y) face velocity is 0; its tangential (x) speed
    is 1 and must appear in the ghost ring of ``up`` exactly at the wall."""
    g = _unit()
    u = g.zeros_u()
    v = g.zeros_v()
    apply_velocity_bc(u, v, walls_bc(velocity=(1.0, 0.0)))
    np.testing.assert_allclose(v[-1, :], 0.0)

    up, _vp = pad_velocity(u, v, walls_bc(velocity=(1.0, 0.0)))
    # the top row of up (ghost row -1, i.e. y = (ny+1/2)h) mirrors u about the lid speed 1
    np.testing.assert_allclose(up[-1, 1:-1], 2.0 * 1.0 - up[-2, 1:-1])
    # interpolating exactly on the wall y = ly must return the lid speed
    y_wall = g.ly
    x = np.linspace(0.0, g.lx, 11)
    sampled = bilinear(up, x, np.full_like(x, y_wall), -g.h, -0.5 * g.h, g.h)
    np.testing.assert_allclose(sampled, 1.0, atol=1e-12)


def test_no_slip_tangential_ghost_extinguishes_wall_velocity() -> None:
    """With a stationary wall the interpolated tangential velocity *on* the wall is zero."""
    g = _unit()
    u = g.zeros_u() + 0.7  # uniform slip would be unphysical at the wall
    v = g.zeros_v()
    up, _ = pad_velocity(u, v, walls_bc())
    y_wall = 0.0
    x = np.linspace(0.0, g.lx, 9)
    sampled = bilinear(up, x, np.full_like(x, y_wall), -g.h, -0.5 * g.h, g.h)
    np.testing.assert_allclose(sampled, 0.0, atol=1e-12)


def test_free_slip_normal_velocity_zero() -> None:
    g = _unit()
    u = np.ones(g.shape_u)
    v = np.ones(g.shape_v)
    free = SideBC(Kind.FREE_SLIP)
    bc = BoundaryConditions(free, free, free, free)
    apply_velocity_bc(u, v, bc)
    np.testing.assert_allclose(u[:, 0], 0.0)
    np.testing.assert_allclose(v[-1, :], 0.0)


# --------------------------------------------------------------------------------------
# inflow / outflow
# --------------------------------------------------------------------------------------
def test_inflow_sets_normal_velocity_into_domain() -> None:
    g = _unit()
    u = g.zeros_u()
    v = g.zeros_v()
    bc = BoundaryConditions(
        left=SideBC(Kind.INFLOW, velocity=(0.5, 0.0)),
        right=SideBC(Kind.OUTFLOW),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.NO_SLIP),
    )
    apply_velocity_bc(u, v, bc)
    np.testing.assert_allclose(u[:, 0], 0.5)
    # outflow face is left alone
    np.testing.assert_allclose(u[:, -1], 0.0)


def test_inflow_from_right_points_leftwards() -> None:
    g = _unit()
    u = g.zeros_u()
    v = g.zeros_v()
    bc = BoundaryConditions(
        left=SideBC(Kind.OUTFLOW),
        right=SideBC(Kind.INFLOW, velocity=(0.5, 0.0)),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.NO_SLIP),
    )
    apply_velocity_bc(u, v, bc)
    np.testing.assert_allclose(u[:, -1], -0.5)


def test_inflow_sets_scalar_value() -> None:
    g = _unit()
    field = np.ones(g.shape_cell)
    bc = BoundaryConditions(
        left=SideBC(Kind.INFLOW, velocity=(1.0, 0.0), scalar=7.0),
        right=SideBC(Kind.OUTFLOW),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.NO_SLIP),
    )
    p = pad_scalar(field, bc)
    np.testing.assert_allclose(p[1:-1, 0], 7.0)


def test_outflow_scalar_uses_zero_gradient() -> None:
    g = _unit()
    X, _ = g.cell_coords()
    field = np.exp(-5.0 * X)
    bc = BoundaryConditions(
        left=SideBC(Kind.INFLOW, velocity=(1.0, 0.0), scalar=1.0),
        right=SideBC(Kind.OUTFLOW),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.NO_SLIP),
    )
    p = pad_scalar(field, bc)
    np.testing.assert_allclose(p[1:-1, -1], field[:, -1])


# --------------------------------------------------------------------------------------
# periodic wrapping in the ghosts
# --------------------------------------------------------------------------------------
def test_periodic_padding_wraps_both_components() -> None:
    g = Grid(6, 6, lx=1.0)
    rng = np.random.default_rng(2)
    u = rng.uniform(-1, 1, g.shape_u)
    v = rng.uniform(-1, 1, g.shape_v)
    up, vp = pad_velocity(u, v, periodic_bc())
    ny, nx = g.ny, g.nx
    # up index i is at x = (i-1)h and vp index i at x = (i-1/2)h, so the periodic wrap is
    # exactly nx *columns* for both: ghost i=0 is interior i=nx, ghost i=nx+2 is i=2.
    np.testing.assert_allclose(up[:, 0], up[:, nx])
    np.testing.assert_allclose(up[:, nx + 2], up[:, 2])
    np.testing.assert_allclose(vp[:, 0], vp[:, nx])
    np.testing.assert_allclose(vp[:, nx + 1], vp[:, 1])
    # up rows are at y = (j-1/2)h (period ny rows); vp rows at y = (j-1)h (period ny rows)
    np.testing.assert_allclose(up[0, :], up[ny, :])
    np.testing.assert_allclose(up[ny + 1, :], up[1, :])
    np.testing.assert_allclose(vp[0, :], vp[ny, :])
    np.testing.assert_allclose(vp[ny + 2, :], vp[2, :])


def test_periodic_scalar_wrap() -> None:
    g = Grid(5, 4, lx=1.0)
    X, Y = g.cell_coords()
    field = np.sin(2 * np.pi * X) + 0.5 * Y
    p = pad_scalar(field, periodic_bc())
    np.testing.assert_allclose(p[1:-1, 0], field[:, -1])
    np.testing.assert_allclose(p[1:-1, -1], field[:, 0])
    np.testing.assert_allclose(p[0, 1:-1], field[-1, :])
    np.testing.assert_allclose(p[-1, 1:-1], field[0, :])


def test_periodic_sampling_is_seamless() -> None:
    """Sampling just inside x=0 and just inside x=lx gives values that match."""
    g = Grid(16, 16, lx=1.0)
    u, v = taylor_green(g)
    up, _ = pad_velocity(u, v, periodic_bc())
    eps = 1e-9
    left = bilinear(up, np.array([0.0 + eps]), np.array([0.5]), -g.h, -0.5 * g.h, g.h)
    right = bilinear(up, np.array([g.lx - eps]), np.array([0.5]), -g.h, -0.5 * g.h, g.h)
    np.testing.assert_allclose(left, right, atol=1e-6)


# --------------------------------------------------------------------------------------
# wall-bounded divergence-free field built from a vanishing streamfunction
# --------------------------------------------------------------------------------------
def test_streamfunction_curl_with_zero_boundary_is_wall_tangential() -> None:
    g = Grid(16, 16, lx=1.0)
    X, Y = g.corner_coords()
    psi = np.sin(np.pi * X) * np.sin(np.pi * Y)  # vanishes on all four walls
    u, v = curl_streamfunction(psi, g.h)
    np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[-1, :], 0.0, atol=1e-12)


def test_sidebc_helpers() -> None:
    s = SideBC(Kind.NO_SLIP, velocity=(0.3, 0.4))
    assert s.tangential(Side.TOP) == pytest.approx(0.3)
    assert s.tangential(Side.LEFT) == pytest.approx(0.4)
    assert SideBC(Kind.INFLOW, velocity=(0.7, 0.0)).normal_into_domain(Side.LEFT) == pytest.approx(0.7)
    assert SideBC(Kind.INFLOW, velocity=(0.7, 0.0)).normal_into_domain(Side.RIGHT) == pytest.approx(-0.7)


def test_free_slip_copies_tangential_and_reflects_normal_ghosts() -> None:
    g = _unit()
    u = np.ones(g.shape_u)
    v = np.ones(g.shape_v)
    free = SideBC(Kind.FREE_SLIP)
    bc = BoundaryConditions(free, free, free, free)
    apply_velocity_bc(u, v, bc)
    up, vp = pad_velocity(u, v, bc)
    x = np.linspace(g.h, g.lx - g.h, 7)
    y = np.linspace(g.h, g.ly - g.h, 7)
    # tangential v at the left wall and u at the bottom wall keep their interior value
    np.testing.assert_allclose(bilinear(vp, np.zeros_like(y), y, -0.5*g.h, -g.h, g.h), 1.0)
    np.testing.assert_allclose(bilinear(up, x, np.zeros_like(x), -g.h, -0.5*g.h, g.h), 1.0)
    # normal components are zero on the wall itself
    np.testing.assert_allclose(bilinear(up, np.zeros_like(y), y, -g.h, -0.5*g.h, g.h), 0.0)
    np.testing.assert_allclose(bilinear(vp, x, np.zeros_like(x), -0.5*g.h, -g.h, g.h), 0.0)
