"""Milestone 1: grid indexing, coordinates, interpolation, divergence, energy."""

from __future__ import annotations

import numpy as np
import pytest

from mackerel.analytic import (
    checkerboard_pressure_field,
    solid_body_rotation,
    taylor_green,
    taylor_green_velocity,
)
from mackerel.diagnostics import (
    divergence_field,
    divergence_norms_on_grid,
    enstrophy,
    kinetic_energy,
    max_cfl,
    total_scalar,
    vorticity_field,
)
from mackerel.grid import Grid, bilinear, curl_streamfunction, sample_cell


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def test_grid_geometry() -> None:
    g = Grid(8, 4, lx=4.0)
    assert g.h == pytest.approx(0.5)
    assert g.ly == pytest.approx(2.0)
    assert g.shape_cell == (4, 8)
    assert g.shape_u == (4, 9)
    assert g.shape_v == (5, 8)
    assert g.shape_corner == (5, 9)
    assert g.cell_area == pytest.approx(0.25)


def test_field_shapes_match_layout() -> None:
    g = Grid(7, 5, lx=1.4)
    assert g.zeros_cell().shape == (5, 7)
    assert g.zeros_u().shape == (5, 8)
    assert g.zeros_v().shape == (6, 7)
    assert g.zeros_corner().shape == (6, 8)
    X, _Y = g.cell_coords()
    assert X.shape == (5, 7)


def test_cell_coordinates_are_half_offsets() -> None:
    g = Grid(4, 3, lx=1.0)
    np.testing.assert_allclose(g.x_centers(), [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(g.x_faces(), [0.0, 0.25, 0.5, 0.75, 1.0])
    Xu, Yu = g.u_coords()
    np.testing.assert_allclose(Xu[0], g.x_faces())
    np.testing.assert_allclose(Yu[:, 0], [0.125, 0.375, 0.625])
    Xv, Yv = g.v_coords()
    np.testing.assert_allclose(Yv[:, 0], g.y_faces())
    np.testing.assert_allclose(Xv[0], [0.125, 0.375, 0.625, 0.875])


def test_grid_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        Grid(0, 4)
    with pytest.raises(ValueError):
        Grid(4, 4, lx=-1.0)


# --------------------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------------------
def test_bilinear_exact_on_bilinear_function() -> None:
    """Bilinear interpolation must be exact on ``a + b x + c y + d x y``."""
    g = Grid(9, 7, lx=1.8)
    h = g.h
    X, Y = g.cell_coords()
    a, b, c, d = 0.3, -1.7, 2.2, 0.9
    field = a + b * X + c * Y + d * X * Y
    rng = np.random.default_rng(0)
    # points inside the domain but not on grid nodes
    xs = rng.uniform(0.5 * h, g.lx - 0.5 * h, 200)
    ys = rng.uniform(0.5 * h, g.ly - 0.5 * h, 200)
    got = sample_cell(field, g, xs, ys)
    want = a + b * xs + c * ys + d * xs * ys
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_bilinear_reproduces_nodes() -> None:
    g = Grid(6, 6, lx=1.0)
    X, Y = g.cell_coords()
    field = np.sin(3.0 * X) * np.cos(2.0 * Y)
    got = sample_cell(field, g, X, Y)
    np.testing.assert_allclose(got, field, atol=1e-13)


def test_bilinear_second_order_on_smooth_field() -> None:
    """Error of bilinear interpolation on ``sin(2 pi x) sin(2 pi y)`` must be O(h^2)."""
    errs: list[float] = []
    hs: list[float] = []
    rng = np.random.default_rng(1)
    for n in (16, 32, 64, 128):
        g = Grid(n, n, lx=1.0)
        X, Y = g.cell_coords()
        field = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        xs = rng.uniform(0.0, 1.0, 4000)
        ys = rng.uniform(0.0, 1.0, 4000)
        got = sample_cell(field, g, xs, ys, clamp=False)
        want = np.sin(2 * np.pi * xs) * np.sin(2 * np.pi * ys)
        errs.append(float(np.sqrt(np.mean((got - want) ** 2))))
        hs.append(g.h)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order == pytest.approx(2.0, abs=0.2), (hs, errs, order)


def test_interpolation_ignores_ci_shape() -> None:
    """Scalar sampling must not care about argument shape (used by scalar advection)."""
    g = Grid(8, 8)
    X, Y = g.cell_coords()
    field = X + 2 * Y
    got = bilinear(field, 0.5, 0.5, 0.5 * g.h, 0.5 * g.h, g.h)
    assert float(got) == pytest.approx(1.5, abs=1e-12)


# --------------------------------------------------------------------------------------
# divergence of test fields
# --------------------------------------------------------------------------------------
def test_discrete_divergence_of_curl_is_zero() -> None:
    """A MAC field built as the discrete curl of a streamfunction is exactly div-free."""
    g = Grid(17, 11, lx=1.7)
    X, Y = g.corner_coords()
    psi = np.sin(3.0 * X) * np.sin(2.0 * Y) + 0.3 * X**2 * Y
    u, v = curl_streamfunction(psi, g.h)
    div = divergence_field(u, v, g)
    assert np.max(np.abs(div)) < 1e-12


def test_analytic_divergence_free_field_has_second_order_divergence() -> None:
    """Discretely differencing a *continuum* div-free field leaves an O(h^2) residual.

    The field is the continuum vortex with streamfunction ``psi = exp(-r^2 / 2 s^2)``,
    i.e. ``u = (y-yc)/s^2 psi`` and ``v = -(x-xc)/s^2 psi``, for which
    ``du/dx + dv/dy = 0`` identically in the continuum but not for the discrete operator.
    (Taylor-Green is a bad choice here: its centred differences cancel exactly, so the
    residual sits at machine precision and no rate can be measured.)
    """
    errs: list[float] = []
    hs: list[float] = []
    s2 = 0.1**2
    for n in (16, 32, 64, 128):
        g = Grid(n, n, lx=1.0)
        Xu, Yu = g.u_coords()
        psi_u = np.exp(-((Xu - 0.5) ** 2 + (Yu - 0.5) ** 2) / (2 * s2))
        u = (Yu - 0.5) / s2 * psi_u
        Xv, Yv = g.v_coords()
        psi_v = np.exp(-((Xv - 0.5) ** 2 + (Yv - 0.5) ** 2) / (2 * s2))
        v = -(Xv - 0.5) / s2 * psi_v
        errs.append(float(np.max(np.abs(divergence_field(u, v, g)))))
        hs.append(g.h)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order == pytest.approx(2.0, abs=0.25), (hs, errs, order)


def test_rotation_divergence_exactly_zero() -> None:
    g = Grid(9, 9, lx=1.0)
    u, v = solid_body_rotation(g, omega=2.5, center=(0.4, 0.6))
    _, l2 = divergence_norms_on_grid(divergence_field(u, v, g), g)
    assert l2 < 1e-12


# --------------------------------------------------------------------------------------
# staggering vs checkerboard
# --------------------------------------------------------------------------------------
def test_staggered_grid_kills_checkerboard_null_space() -> None:
    """The ``(-1)^(i+j)`` pressure mode is invisible to a collocated gradient.

    On a collocated (cell-centred) grid the centred difference of ``(-1)^(i+j)`` is zero,
    so the mode can grow unchecked.  On a staggered grid the same mode produces a face
    gradient of magnitude ``2A/h`` -- the largest possible -- so it is strongly damped.
    """
    g = Grid(16, 16)
    p = checkerboard_pressure_field(g, amplitude=1.0)
    # collocated: (p[i+1] - p[i-1]) / (2h) on interior cells
    gx_col = np.zeros_like(p)
    gx_col[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * g.h)
    assert np.max(np.abs(gx_col[:, 1:-1])) == pytest.approx(0.0, abs=1e-14)

    # staggered: (p[i] - p[i-1]) / h on interior faces
    gx_stag = np.zeros((g.ny, g.nx + 1))
    gx_stag[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / g.h
    np.testing.assert_allclose(np.abs(gx_stag[:, 1:-1]), 2.0 / g.h)


# --------------------------------------------------------------------------------------
# energy / enstrophy convergence
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [16, 32, 64])
def test_taylor_green_energy_matches_analytic_quadrature(n: int) -> None:
    """Midpoint-rule energy of the sampled Taylor-Green field is the analytic 0.25 + O(h^2).

    The measured leading coefficient is about ``3.3 h^2`` (0.0126 at h=1/16, 0.0032 at
    h=1/32, 0.00080 at h=1/64), caused by the centred-difference and face-averaging errors
    in the corner-curl construction; the bound below is ``4 h^2``.
    """
    g = Grid(n, n, lx=1.0)
    u, v = taylor_green(g)
    e = kinetic_energy(u, v, g)
    assert abs(e - 0.25) < 4.0 * g.h**2


def test_taylor_green_energy_converges_second_order() -> None:
    errs: list[float] = []
    hs: list[float] = []
    for n in (8, 16, 32, 64, 128):
        g = Grid(n, n, lx=1.0)
        u, v = taylor_green(g)
        errs.append(abs(kinetic_energy(u, v, g) - 0.25))
        hs.append(g.h)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order == pytest.approx(2.0, abs=0.25), (hs, errs, order)


def test_taylor_green_enstrophy_converges_second_order() -> None:
    """Exact enstrophy of the unit-amplitude TG field is 2 k^2 E = 2*(2pi)^2*0.25 = 2 pi^2."""
    exact = 2.0 * (2 * np.pi) ** 2 * 0.25
    errs: list[float] = []
    hs: list[float] = []
    for n in (16, 32, 64, 128):
        g = Grid(n, n, lx=1.0)
        u, v = taylor_green(g)
        errs.append(abs(enstrophy(u, v, g) - exact))
        hs.append(g.h)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order == pytest.approx(2.0, abs=0.3), (hs, errs, order)


def test_taylor_green_velocity_error_second_order() -> None:
    errs: list[float] = []
    hs: list[float] = []
    for n in (16, 32, 64):
        g = Grid(n, n, lx=1.0)
        u, v = taylor_green(g)
        Xu, Yu = g.u_coords()
        ue, _ = taylor_green_velocity(g, Xu, Yu)
        Xv, Yv = g.v_coords()
        _, ve = taylor_green_velocity(g, Xv, Yv)
        e = float(np.sqrt(np.mean((u - ue) ** 2) + np.mean((v - ve) ** 2)))
        errs.append(e)
        hs.append(g.h)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order == pytest.approx(2.0, abs=0.2), (hs, errs, order)


# --------------------------------------------------------------------------------------
# misc diagnostics
# --------------------------------------------------------------------------------------
def test_vorticity_of_rotation_is_2_omega() -> None:
    g = Grid(20, 20, lx=1.0)
    omega = 1.5
    u, v = solid_body_rotation(g, omega=omega)
    w = vorticity_field(u, v, g)[1:-1, 1:-1]
    np.testing.assert_allclose(w, 2.0 * omega, rtol=0, atol=1e-12)


def test_vorticity_sign_convention() -> None:
    """Counter-clockwise rotation has positive vorticity."""
    g = Grid(10, 10, lx=1.0)
    u, v = solid_body_rotation(g, omega=1.0)
    assert float(np.mean(vorticity_field(u, v, g))) > 0.0


def test_total_scalar_midpoint_rule() -> None:
    g = Grid(8, 8, lx=1.0)
    field = g.zeros_cell() + 2.0
    assert total_scalar(field, g) == pytest.approx(2.0)  # area 1


def test_max_cfl_scaling() -> None:
    g = Grid(10, 10, lx=1.0)
    u = g.zeros_u() + 3.0
    v = g.zeros_v()
    assert max_cfl(u, v, g, dt=0.01) == pytest.approx(0.01 * 3.0 / g.h)
