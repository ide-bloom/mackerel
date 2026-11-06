"""Tests for the implicit viscous step (:mod:`mackerel.diffuse`)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from mackerel.analytic import taylor_green
from mackerel.diagnostics import kinetic_energy
from mackerel.diffuse import (
    GHOST,
    NEUMANN,
    PERIODIC,
    _System,
    _velocity_system,
    diffuse,
    diffuse_scalar,
)
from mackerel.grid import (
    BoundaryConditions,
    Grid,
    Kind,
    SideBC,
    apply_velocity_bc,
    periodic_bc,
    walls_bc,
)


def channel_bc() -> BoundaryConditions:
    """Periodic in x, no-slip walls above and below (plane Poiseuille geometry)."""
    p = SideBC(Kind.PERIODIC)
    w = SideBC(Kind.NO_SLIP)
    return BoundaryConditions(p, p, w, w)


# --------------------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------------------
def test_periodic_operator_matches_five_point_laplacian() -> None:
    """On a reduced periodic array ``A x = x - alpha L x`` exactly, with ``L`` wrapping."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 12))
    alpha = 0.037
    bc = periodic_bc()
    sys = _velocity_system("u", x.shape, bc, alpha)
    lap = (
        np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        + np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        - 4.0 * x
    )
    assert np.allclose(sys.apply(x), x - alpha * lap, atol=1e-14)


def test_neumann_operator_drops_outside_coupling() -> None:
    """Zero-gradient sides remove the neighbour *and* its diagonal contribution."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((9, 9))
    alpha = 0.2
    sys = _System(alpha=alpha, x_mode=NEUMANN, y_mode=NEUMANN)
    # Reference: pad with zero-gradient copies and apply the usual 5-point Laplacian.
    padded = np.zeros((11, 11))
    padded[1:-1, 1:-1] = x
    padded[0, 1:-1] = x[0, :]
    padded[-1, 1:-1] = x[-1, :]
    padded[1:-1, 0] = x[:, 0]
    padded[1:-1, -1] = x[:, -1]
    lap = (
        padded[2:, 1:-1] + padded[:-2, 1:-1]
        + padded[1:-1, 2:] + padded[1:-1, :-2] - 4.0 * x
    )
    assert np.allclose(sys.apply(x), x - alpha * lap, atol=1e-14)


def test_operator_is_symmetric_and_positive_definite() -> None:
    """The implicit system must be SPD, otherwise PCG is not applicable at all."""
    for bc in (periodic_bc(), walls_bc(), channel_bc()):
        for component in ("u", "v"):
            shape = Grid(6, 5).shape_u if component == "u" else Grid(6, 5).shape_v
            red = (
                shape[0] - 1
                if component == "v" and bc.is_y_periodic
                else (shape[1] - 1 if component == "u" and bc.is_x_periodic else shape[1])
            )
            ny = shape[0] - 1 if component == "v" and bc.is_y_periodic else shape[0]
            n = ny * red
            sys = _velocity_system(component, (ny, red), bc, 0.05)
            mat = np.zeros((n, n))
            for k in range(n):
                e = np.zeros(n)
                e[k] = 1.0
                mat[:, k] = sys.apply(e.reshape(ny, red)).ravel()
            assert np.allclose(mat, mat.T, atol=1e-13), (component, bc)
            # Prescribed rows/columns are eliminated (zero), so inspect the actual unknowns.
            free = ~sys.fixed.ravel() if sys.fixed.size else np.ones(n, dtype=bool)
            reduced = mat[np.ix_(free, free)]
            assert np.min(np.linalg.eigvalsh(reduced)) > 0.0, (component, bc)


# --------------------------------------------------------------------------------------
# physical behaviour
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("nu", [1e-3, 1e-2])
def test_diffusion_matches_taylor_green_decay(nu: float) -> None:
    """Periodic diffusion of Taylor-Green must follow ``E(t) = E0 exp(-4 k^2 nu t)``.

    Backward Euler damps slightly more than the exact exponential (factor
    ``1/(1 + 4 k^2 nu dt)`` per step instead of ``exp(-4 k^2 nu dt)``), so the agreement is
    first order in ``dt``; at ``dt = 2.5e-3`` the measured relative energy error is below
    1.5e-3 (the spatial five-point eigenvalue contributes a further small error).
    """
    grid = Grid(64, 64, lx=1.0)
    u, v = taylor_green(grid)
    bc = periodic_bc()
    tau = 0.5
    dt = 2.5e-3
    e0 = kinetic_energy(u, v, grid)
    for _ in range(round(tau / dt)):
        res = diffuse(u, v, grid, bc, nu, dt)
        u, v = res.u, res.v
    exact = e0 * np.exp(-4.0 * (2.0 * np.pi) ** 2 * nu * tau)
    assert kinetic_energy(u, v, grid) == pytest.approx(exact, rel=1.5e-3)


def test_diffusion_keeps_periodic_seam_identical() -> None:
    """The duplicated seam face must stay duplicated; otherwise the field stops being periodic."""
    grid = Grid(16, 16, lx=1.0)
    u, v = taylor_green(grid)
    res = diffuse(u, v, grid, periodic_bc(), 1e-2, 0.01)
    assert np.array_equal(res.u[:, 0], res.u[:, -1])
    assert np.array_equal(res.v[0, :], res.v[-1, :])


def test_diffusion_monotone_in_time() -> None:
    """Viscosity never adds energy in a periodic box."""
    grid = Grid(32, 32, lx=1.0)
    u, v = taylor_green(grid)
    bc = periodic_bc()
    energies = []
    for _ in range(20):
        res = diffuse(u, v, grid, bc, 1e-2, 5e-3)
        u, v = res.u, res.v
        energies.append(kinetic_energy(u, v, grid))
    assert all(b <= a + 1e-12 for a, b in pairwise(energies))


def test_uniform_flow_is_a_steady_viscous_solution() -> None:
    """A constant periodic field has ``L u = 0``, so diffusion must leave it unchanged."""
    grid = Grid(32, 32, lx=1.0)
    u = np.full(grid.shape_u, 0.3)
    v = np.full(grid.shape_v, -0.2)
    bc = periodic_bc()
    res = diffuse(u, v, grid, bc, 1e-2, 0.01)
    assert np.max(np.abs(res.u - u)) < 1e-12
    assert np.max(np.abs(res.v - v)) < 1e-12


def test_no_slip_wall_slows_the_tangential_flow() -> None:
    """A shear flow next to no-slip walls must lose speed near them, not gain it."""
    grid = Grid(16, 16, lx=1.0)
    u = np.ones(grid.shape_u)
    v = np.zeros(grid.shape_v)
    bc = channel_bc()
    apply_velocity_bc(u, v, bc)
    res = diffuse(u, v, grid, bc, 0.1, 0.05)
    assert res.u[0, 1] < u[0, 1]
    assert res.u[-1, 1] < u[-1, 1]
    assert res.u[grid.ny // 2, 1] > res.u[0, 1]


def test_free_slip_leaves_uniform_flow_unchanged() -> None:
    """Zero-gradient walls have no shear, so a uniform field is an exact steady state."""
    grid = Grid(16, 16, lx=1.0)
    u = np.ones(grid.shape_u)
    v = np.zeros(grid.shape_v)
    p = SideBC(Kind.PERIODIC)
    f = SideBC(Kind.FREE_SLIP)
    bc = BoundaryConditions(p, p, f, f)
    res = diffuse(u, v, grid, bc, 0.1, 0.05)
    assert np.max(np.abs(res.u - u)) < 1e-12


def test_ghost_side_declares_a_dirichlet_wall() -> None:
    """A no-slip side is the GHOST mode and reports the wall speed as its source term."""
    bc = channel_bc()
    sys = _velocity_system("u", (8, 8), bc, 0.1)
    assert sys.y_mode == GHOST
    assert sys.x_mode == PERIODIC
    assert sys.y_low == 0.0 and sys.y_high == 0.0

    lid = BoundaryConditions(
        SideBC(Kind.PERIODIC), SideBC(Kind.PERIODIC),
        SideBC(Kind.NO_SLIP), SideBC(Kind.NO_SLIP, velocity=(1.0, 0.0)),
    )
    sys_lid = _velocity_system("u", (8, 8), lid, 0.1)
    assert sys_lid.y_high == 1.0
    # the lid enters the right-hand side of the top row only
    src = sys_lid.source((8, 8))
    assert src[-1, 0] > 0.0 and src[0, 0] == 0.0


def test_zero_viscosity_is_exactly_identity() -> None:
    grid = Grid(8, 8, lx=1.0)
    u, v = taylor_green(grid)
    res = diffuse(u, v, grid, periodic_bc(), 0.0, 0.1)
    assert np.array_equal(res.u, u) and np.array_equal(res.v, v)
    assert res.iterations == 0


# --------------------------------------------------------------------------------------
# scalars
# --------------------------------------------------------------------------------------
def test_scalar_diffusion_decays_a_sine_wave() -> None:
    """A single periodic mode decays as ``exp(-k^2 nu t)`` (up to the backward-Euler factor)."""
    grid = Grid(64, 64, lx=1.0)
    k = 2.0 * np.pi
    X, Y = grid.cell_coords()
    field = np.sin(k * X) * np.cos(k * Y)
    nu, dt, tau = 0.01, 5e-3, 0.5
    f = field
    for _ in range(round(tau / dt)):
        f = diffuse_scalar(f, grid, periodic_bc(), nu, dt)
    exact = np.exp(-2.0 * k**2 * nu * tau)
    measured = float(np.sum(f * field) / np.sum(field * field))
    assert measured == pytest.approx(exact, rel=5e-3)


def test_scalar_diffusion_conserves_a_mean() -> None:
    """Zero-gradient walls let nothing out, so the integral of the scalar is preserved."""
    grid = Grid(24, 24, lx=1.0)
    rng = np.random.default_rng(3)
    f = rng.standard_normal(grid.shape_cell)
    out = diffuse_scalar(f, grid, walls_bc(), 0.05, 0.02)
    # Measured round-off/PCG residual at tol=1e-10 is 3.8e-9 in the integral.
    assert float(np.sum(out)) == pytest.approx(float(np.sum(f)), abs=1e-8)
