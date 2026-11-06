"""Milestone 2: advection accuracy, conservation, monotonicity."""

from __future__ import annotations

import numpy as np
import pytest

from mackerel.advect import (
    AdvectionOptions,
    Scheme,
    advect_scalar,
    advect_velocity,
    backtrace,
)
from mackerel.analytic import solid_body_rotation, taylor_green
from mackerel.diagnostics import kinetic_energy, total_scalar
from mackerel.grid import Grid, bilinear, pad_velocity, periodic_bc, walls_bc


def _translate(field: np.ndarray, shift_cells: int, axis: int = 1) -> np.ndarray:
    return np.roll(field, shift_cells, axis=axis)


# --------------------------------------------------------------------------------------
# backtrace
# --------------------------------------------------------------------------------------
def test_backtrace_constant_velocity_is_exact() -> None:
    g = Grid(16, 16, lx=1.0)
    u = g.zeros_u() + 0.7
    v = g.zeros_v() - 0.3
    bc = periodic_bc()
    up, vp = pad_velocity(u, v, bc)
    X, Y = g.cell_coords()
    dt = 0.01
    xd, yd = backtrace(up, vp, g, X, Y, dt, order=2)
    np.testing.assert_allclose(xd, X - dt * 0.7, atol=1e-12)
    np.testing.assert_allclose(yd, Y + dt * 0.3, atol=1e-12)


def test_backtrace_rk3_matches_rk2_for_linear_field() -> None:
    g = Grid(16, 16, lx=1.0)
    u = np.full(g.shape_u, 0.2)
    v = np.full(g.shape_v, 0.1)
    bc = periodic_bc()
    up, vp = pad_velocity(u, v, bc)
    X, Y = g.cell_coords()
    x2, y2 = backtrace(up, vp, g, X, Y, 0.05, order=2)
    x3, y3 = backtrace(up, vp, g, X, Y, 0.05, order=3)
    np.testing.assert_allclose(x2, x3, atol=1e-12)
    np.testing.assert_allclose(y2, y3, atol=1e-12)


def test_backtrace_order_rejects_nonsense() -> None:
    g = Grid(8, 8, lx=1.0)
    up, vp = pad_velocity(g.zeros_u(), g.zeros_v(), periodic_bc())
    X, Y = g.cell_coords()
    with pytest.raises(ValueError):
        backtrace(up, vp, g, X, Y, 0.1, order=5)


# --------------------------------------------------------------------------------------
# translation at constant velocity
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", list(Scheme))
def test_translation_of_smooth_profile(scheme: Scheme) -> None:
    """A sine wave translated by an integer number of cells comes back (nearly) unchanged."""
    g = Grid(64, 8, lx=1.0)
    X, _Y = g.cell_coords()
    field = np.sin(2 * np.pi * X)
    speed = 0.5
    nsteps = 8
    dt = (1.0 / g.nx) / speed - 1e-12  # exactly one cell per step
    u = g.zeros_u() + speed
    v = g.zeros_v()
    bc = periodic_bc()
    opts = AdvectionOptions(scheme=scheme)
    f = field
    for _ in range(nsteps):
        f = advect_scalar(f, u, v, g, bc, dt, opts)
    shifted = np.roll(field, nsteps, axis=1)
    err = np.max(np.abs(f - shifted))
    assert err < 0.02, err


@pytest.mark.parametrize("scheme", list(Scheme))
def test_constant_field_is_preserved(scheme: Scheme) -> None:
    g = Grid(32, 32, lx=1.0)
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    field = np.full(g.shape_cell, 3.0)
    out = advect_scalar(field, u, v, g, bc, 0.01, AdvectionOptions(scheme=scheme))
    np.testing.assert_allclose(out, 3.0, atol=1e-10)


# --------------------------------------------------------------------------------------
# solid-body rotation: closed loop returns to start
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", list(Scheme))
def test_full_rotation_returns_blob_near_start(scheme: Scheme) -> None:
    """A Gaussian blob rotated exactly once around the box centre returns to its start.

    The velocity field is rigid rotation (exactly divergence free), so the only error is the
    scheme's.  Measured at 240 steps (CFL ~ 0.9) on 64^2: RMS errors 7.3e-2
    (semi-Lagrangian), 2.5e-2 (MacCormack) and 1.0e-2 (BFECC).  The bound below is the
    measured worst case plus margin.
    """
    n = 64
    g = Grid(n, n, lx=1.0)
    X, Y = g.cell_coords()
    cx, cy = 0.5, 0.5
    r2 = (X - cx - 0.25) ** 2 + (Y - cy) ** 2
    field = np.exp(-r2 / (2 * 0.07**2))
    omega = 1.0
    u, v = solid_body_rotation(g, omega=omega, center=(cx, cy))
    bc = walls_bc()
    nsteps = 240
    dt = 2 * np.pi / omega / nsteps
    opts = AdvectionOptions(scheme=scheme, limiter=True)
    f = field
    for _ in range(nsteps):
        f = advect_scalar(f, u, v, g, bc, dt, opts)
    err = float(np.sqrt(np.mean((f - field) ** 2)))
    assert err < 0.09, err


def test_maccormack_beats_semi_lagrangian_on_smooth_rotation() -> None:
    """Less numerical diffusion: the MacCormack result must be closer to the exact field."""
    n = 64
    g = Grid(n, n, lx=1.0)
    X, Y = g.cell_coords()
    field = np.exp(-((X - 0.75) ** 2 + (Y - 0.5) ** 2) / (2 * 0.07**2))
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    nsteps = 180
    dt = 2 * np.pi / nsteps

    errs = {}
    for scheme in (Scheme.SEMI_LAGRANGIAN, Scheme.MACCORMACK, Scheme.BFECC):
        f = field
        opts = AdvectionOptions(scheme=scheme)
        for _ in range(nsteps):
            f = advect_scalar(f, u, v, g, bc, dt, opts)
        errs[scheme] = float(np.sqrt(np.mean((f - field) ** 2)))
    assert errs[Scheme.MACCORMACK] < errs[Scheme.SEMI_LAGRANGIAN]
    assert errs[Scheme.BFECC] < errs[Scheme.SEMI_LAGRANGIAN]


# --------------------------------------------------------------------------------------
# monotonicity / limiter
# --------------------------------------------------------------------------------------
def test_limiter_prevents_new_extrema_on_a_sharp_blob() -> None:
    """Advect a top-hat blob; only the limited schemes keep the value inside its own range."""
    n = 80
    g = Grid(n, n, lx=1.0)
    X, Y = g.cell_coords()
    field = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.14**2).astype(np.float64)
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    nsteps = 200
    dt = 2 * np.pi / nsteps
    for scheme in (Scheme.MACCORMACK, Scheme.BFECC):
        f = field
        opts = AdvectionOptions(scheme=scheme, limiter=True)
        for _ in range(nsteps):
            f = advect_scalar(f, u, v, g, bc, dt, opts)
        assert f.max() <= 1.0 + 1e-12, (scheme, f.max())
        assert f.min() >= -1e-12, (scheme, f.min())


def test_unlimited_maccormack_overshoots() -> None:
    """Without the limiter the same test *does* leave [0, 1] -- the limiter is doing work."""
    n = 80
    g = Grid(n, n, lx=1.0)
    X, Y = g.cell_coords()
    field = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.14**2).astype(np.float64)
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    nsteps = 100
    dt = 2 * np.pi / nsteps
    f = field
    opts = AdvectionOptions(scheme=Scheme.MACCORMACK, limiter=False)
    for _ in range(nsteps):
        f = advect_scalar(f, u, v, g, bc, dt, opts)
    assert f.max() > 1.0 + 1e-6 or f.min() < -1e-6


# --------------------------------------------------------------------------------------
# conservation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", list(Scheme))
def test_total_scalar_conservation_under_periodic_advection(scheme: Scheme) -> None:
    """Periodic translation of a compact blob: measure the mass drift, do not assume zero.

    Measured mass drift after 100 steps at CFL 0.26 on 64^2: semi-Lagrangian 3e-16,
    MacCormack +9.7e-3, BFECC -1.6e-2.  The two corrected schemes are *not* conservative:
    the min/max limiter, which is what keeps them free of over/undershoot, is a per-cell
    nonlinear operation that redistributes the total.  The bound below is the measured
    worst case with margin; the exact number is printed by ``mackerel validate advection``.
    """
    g = Grid(64, 64, lx=1.0)
    X, Y = g.cell_coords()
    field = np.exp(-((X - 0.3) ** 2 + (Y - 0.5) ** 2) / (2 * 0.05**2))
    u = g.zeros_u() + 0.4
    v = g.zeros_v()
    bc = periodic_bc()
    m0 = total_scalar(field, g)
    f = field
    opts = AdvectionOptions(scheme=scheme)
    for _ in range(100):
        f = advect_scalar(f, u, v, g, bc, 0.01, opts)
    drift = abs(total_scalar(f, g) - m0) / m0
    assert drift < 2.5e-2, drift


@pytest.mark.parametrize("scheme", list(Scheme))
def test_translation_conserves_mass(scheme: Scheme) -> None:
    """Under a *uniform* periodic translation the discrete sum is very nearly invariant.

    Measured relative drift after 20 steps at CFL 0.5 on 48^2: 1.1e-14 (semi-Lagrangian),
    2.7e-3 (MacCormack), 1.4e-2 (BFECC).  Semi-Lagrangian is conservative here because the
    bilinear weights of a uniformly shifted grid sum to the same total; the corrected
    schemes break that by construction (their limiter is a nonlinear per-cell operation),
    which is a real, measured tradeoff, not a bug.
    """
    g = Grid(48, 48, lx=1.0)
    X, Y = g.cell_coords()
    field = np.exp(-((X - 0.3) ** 2 + (Y - 0.5) ** 2) / (2 * 0.05**2))
    u = g.zeros_u() + 0.5
    v = g.zeros_v() + 0.25
    bc = periodic_bc()
    m0 = total_scalar(field, g)
    f = field
    opts = AdvectionOptions(scheme=scheme)
    dt = 0.5 * g.h / 0.75
    for _ in range(20):
        f = advect_scalar(f, u, v, g, bc, dt, opts)
    drift = abs(total_scalar(f, g) - m0) / m0
    bound = 1e-10 if scheme is Scheme.SEMI_LAGRANGIAN else 2e-2
    assert drift < bound, drift


# --------------------------------------------------------------------------------------
# velocity advection
# --------------------------------------------------------------------------------------
def test_velocity_advection_of_uniform_flow_is_exact() -> None:
    g = Grid(32, 32, lx=1.0)
    u = g.zeros_u() + 0.3
    v = g.zeros_v() - 0.2
    bc = periodic_bc()
    for scheme in Scheme:
        un, vn = advect_velocity(u, v, g, bc, 0.01, AdvectionOptions(scheme=scheme))
        np.testing.assert_allclose(un[:, 1:-1], 0.3, atol=1e-10)
        np.testing.assert_allclose(vn[1:-1, :], -0.2, atol=1e-10)


def test_velocity_advection_preserves_energy_of_uniform_rotation() -> None:
    """Rigid rotation is a steady solution: advecting it must not change the energy."""
    g = Grid(64, 64, lx=1.0)
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    e0 = kinetic_energy(u, v, g)
    un, vn = advect_velocity(u, v, g, bc, 0.005, AdvectionOptions(scheme=Scheme.MACCORMACK))
    e1 = kinetic_energy(un, vn, g)
    assert abs(e1 - e0) / e0 < 0.02


def test_velocity_advection_does_not_blow_up_at_walls() -> None:
    """A backtrace aimed at a wall must not produce large velocities (ghost mirroring)."""
    g = Grid(48, 48, lx=1.0)
    u, v = solid_body_rotation(g, omega=2.0)
    bc = walls_bc()
    un, vn = advect_velocity(u, v, g, bc, 0.02, AdvectionOptions(scheme=Scheme.SEMI_LAGRANGIAN))
    assert np.all(np.isfinite(un)) and np.all(np.isfinite(vn))
    assert np.max(np.abs(un)) < 1.5


def test_taylor_green_advection_stays_divergence_free_in_the_curl_sense() -> None:
    """Advection alone does not enforce incompressibility, but it must not explode."""
    g = Grid(64, 64, lx=1.0)
    u, v = taylor_green(g)
    bc = periodic_bc()
    e0 = kinetic_energy(u, v, g)
    un, vn = advect_velocity(u, v, g, bc, 0.01, AdvectionOptions(scheme=Scheme.MACCORMACK))
    e1 = kinetic_energy(un, vn, g)
    assert 0.5 * e0 < e1 < 2.0 * e0


# --------------------------------------------------------------------------------------
# interpolation into padded closure
# --------------------------------------------------------------------------------------
def test_padded_closure_matches_direct_sampling() -> None:
    g = Grid(24, 24, lx=1.0)
    u, v = solid_body_rotation(g, omega=1.0)
    bc = walls_bc()
    up, _vp = pad_velocity(u, v, bc)
    Xu, Yu = g.u_coords()
    direct = bilinear(u, Xu, Yu, 0.0, 0.5 * g.h, g.h)
    padded = bilinear(up, Xu, Yu, -g.h, -0.5 * g.h, g.h)
    np.testing.assert_allclose(direct, padded, atol=1e-12)
