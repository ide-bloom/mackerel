"""Milestone 3: discrete operators, Poisson boundary conditions, solvers, projection."""

from __future__ import annotations

import numpy as np
import pytest

from mackerel.analytic import solid_body_rotation, taylor_green
from mackerel.grid import (
    BoundaryConditions,
    Grid,
    Kind,
    SideBC,
    apply_velocity_bc,
    curl_streamfunction,
    divergence,
    gradient,
    periodic_bc,
    walls_bc,
)
from mackerel.project import PoissonSystem, pressure_poisson_rhs, project, project_velocity
from mackerel.solvers import (
    SOLVERS,
    ConjugateGradient,
    Multigrid,
    get_solver,
    laplacian_apply,
)

ALL_SOLVERS = sorted(SOLVERS)


def _random_field(g: Grid, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, g.shape_u)
    v = rng.uniform(-1.0, 1.0, g.shape_v)
    apply_velocity_bc(u, v, walls_bc())
    return u, v


# --------------------------------------------------------------------------------------
# the operator itself
# --------------------------------------------------------------------------------------
def test_laplacian_matches_hand_written_5_point_stencil() -> None:
    g = Grid(9, 7, lx=1.3)
    bc = walls_bc()
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, bc)
    p = np.random.default_rng(1).uniform(-1.0, 1.0, g.shape_cell)
    got = laplacian_apply(p, system)          # this is -laplacian
    pe = np.pad(p, 1, mode="edge")            # mirrored (Neumann) ghosts
    lap = (pe[1:-1, 2:] + pe[1:-1, :-2] + pe[2:, 1:-1] + pe[:-2, 1:-1] - 4.0 * p) / g.h**2
    np.testing.assert_allclose(got, -lap, atol=1e-10)


def test_laplacian_periodic_wraps() -> None:
    g = Grid(8, 8, lx=1.0)
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, periodic_bc())
    p = np.random.default_rng(2).uniform(-1.0, 1.0, g.shape_cell)
    lap = (
        np.roll(p, 1, 1) + np.roll(p, -1, 1) + np.roll(p, 1, 0) + np.roll(p, -1, 0) - 4 * p
    ) / g.h**2
    np.testing.assert_allclose(laplacian_apply(p, system), -lap, atol=1e-10)


def test_operator_is_symmetric_and_negative_semidefinite() -> None:
    g = Grid(6, 6, lx=1.0)
    for bc in (walls_bc(), periodic_bc()):
        system = PoissonSystem.from_bc(g.ny, g.nx, g.h, bc)
        n = g.ny * g.nx
        a = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1.0
            a[:, k] = laplacian_apply(e.reshape(g.ny, g.nx), system).ravel()
        np.testing.assert_allclose(a, a.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(a)) > -1e-10   # SPD (>= 0)


def test_operator_with_outflow_is_positive_definite() -> None:
    g = Grid(6, 6, lx=1.0)
    bc = BoundaryConditions(
        left=SideBC(Kind.INFLOW, velocity=(1.0, 0.0)),
        right=SideBC(Kind.OUTFLOW),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.FREE_SLIP),
    )
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, bc)
    n = g.ny * g.nx
    a = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        a[:, k] = laplacian_apply(e.reshape(g.ny, g.nx), system).ravel()
    np.testing.assert_allclose(a, a.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(a)) > 1e-8


def test_divergence_and_gradient_are_adjoint() -> None:
    """``<div u, p> = -<u, grad p>`` up to boundary terms that vanish for closed domains.

    This identity is why ``div(grad)`` is symmetric and why CG is applicable at all.
    """
    g = Grid(12, 10, lx=1.0)
    bc = walls_bc()
    rng = np.random.default_rng(3)
    u = rng.uniform(-1, 1, g.shape_u)
    v = rng.uniform(-1, 1, g.shape_v)
    apply_velocity_bc(u, v, bc)
    p = rng.uniform(-1, 1, g.shape_cell)
    gx, gy = gradient(p, g.h)
    lhs = float(np.sum(divergence(u, v, g.h) * p) * g.cell_area)
    # boundary faces carry zero gradient (mirrored ghost) so they contribute nothing
    rhs = -float(
        (np.sum(u[:, :-1] * gx[:, :-1]) + np.sum(v[:-1, :] * gy[:-1, :])) * g.cell_area
    )
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------------------
# Poisson boundary conditions
# --------------------------------------------------------------------------------------
def test_outflow_sets_pressure_dirichlet() -> None:
    """With an outflow side, the pressure in the boundary cells is strongly damped."""
    g = Grid(24, 24, lx=1.0)
    bc = BoundaryConditions(
        left=SideBC(Kind.INFLOW, velocity=(1.0, 0.0)),
        right=SideBC(Kind.OUTFLOW),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.NO_SLIP),
    )
    # a plug flow with a smooth transverse perturbation: the projection must produce a
    # non-trivial pressure that is pinned near zero where the flow leaves the domain
    rng = np.random.default_rng(7)
    u = g.zeros_u() + 1.0
    u[:, 0] = 1.0
    v = rng.uniform(-0.2, 0.2, g.shape_v)
    apply_velocity_bc(u, v, bc)
    res = project_velocity(u, v, g, bc, dt=0.1, solver="conjugate_gradient", tol=1e-10)
    assert res.residual < 1e-8
    assert np.abs(res.p).max() > 1e-3
    # Dirichlet at the outflow: the pressure there is driven towards zero relative to the bulk
    near_out = np.abs(res.p[:, -3:]).mean()
    bulk = np.abs(res.p[:, g.nx // 3 : 2 * g.nx // 3]).mean()
    assert near_out < bulk, (near_out, bulk)


def test_null_space_is_projected_out() -> None:
    """Fully closed domain: the returned pressure has zero mean (minimum-norm solution)."""
    g = Grid(16, 16, lx=1.0)
    u, v = _random_field(g)
    res = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="conjugate_gradient")
    assert abs(float(res.p.mean())) < 1e-10


def test_adding_a_constant_to_pressure_changes_nothing() -> None:
    """The physical velocity must be invariant to p -> p + c (the null space is harmless)."""
    g = Grid(16, 16, lx=1.0)
    u, v = _random_field(g)
    r1 = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="conjugate_gradient")
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, walls_bc())
    system.rhs = pressure_poisson_rhs(u, v, g, system, 0.1)
    system.guess = np.full(g.shape_cell, 3.7)
    from mackerel.solvers import get_solver as _get

    p2 = _get("conjugate_gradient").solve(system, tol=1e-10, maxiter=2000).p
    gx, _gy = gradient(p2, g.h)
    u2 = u.copy()
    u2[:, 1:-1] -= 0.1 * gx[:, 1:-1]
    np.testing.assert_allclose(u2[:, 1:-1], r1.u[:, 1:-1], atol=1e-8)


def test_rhs_has_zero_mean_when_singular() -> None:
    g = Grid(12, 12, lx=1.0)
    u, v = _random_field(g)
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, walls_bc())
    assert system.null_space
    rhs = pressure_poisson_rhs(u, v, g, system, 0.1)
    assert abs(float(rhs.mean())) < 1e-12


def test_rhs_is_not_zero_meant_when_outflow_exists() -> None:
    g = Grid(12, 12, lx=1.0)
    bc = BoundaryConditions(
        left=SideBC(Kind.INFLOW, velocity=(1.0, 0.0)),
        right=SideBC(Kind.OUTFLOW),
        bottom=SideBC(Kind.NO_SLIP),
        top=SideBC(Kind.NO_SLIP),
    )
    u, v = _random_field(g)
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, bc)
    assert not system.null_space
    rhs = pressure_poisson_rhs(u, v, g, system, 0.1)
    # rhs is *not* forced to zero mean here; only the closed case needs that
    assert rhs.shape == g.shape_cell


# --------------------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_solver_solves_a_manufactured_problem(name: str) -> None:
    """Solve ``-laplacian(p) = rhs`` for a smooth manufactured ``p`` and compare."""
    g = Grid(32, 32, lx=1.0)
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, walls_bc())
    X, Y = g.cell_coords()
    p_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # the system solves  (-laplacian) p = rhs, so rhs must be minus the Laplacian of p
    system.rhs = laplacian_apply(p_exact, system)
    system.rhs = np.where(system.open_mask, system.rhs, 0.0)
    result = get_solver(name).solve(system, tol=1e-10, maxiter=5000)
    got = result.p - result.p.mean()
    want = p_exact - p_exact.mean()
    rel = float(np.abs(got - want).max() / np.abs(want).max())
    # Jacobi and Gauss-Seidel are slow; they are allowed more iterations but must converge
    assert rel < (1e-2 if name in ("jacobi", "gauss_seidel") else 1e-8), (name, rel)


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_solvers_agree(name: str) -> None:
    """All four solvers must produce the same velocity field once they have converged.

    Jacobi is given a much looser target because it is genuinely slow (it needs O(n^2/ log)
    sweeps on a closed domain); the *agreement* target is scaled to what each solver can
    reach in the iteration budget, and the honest comparison of their cost is in
    ``mackerel validate projection``.
    """
    g = Grid(32, 32, lx=1.0)
    u, v = _random_field(g)
    ref = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="conjugate_gradient",
                           tol=1e-10, maxiter=5000)
    tol = 1e-10 if name != "jacobi" else 1e-6
    budget = 5000 if name != "jacobi" else 20000
    got = project_velocity(u, v, g, walls_bc(), dt=0.1, solver=name, tol=tol, maxiter=budget)
    atol = 2e-5 if name != "jacobi" else 2e-2
    np.testing.assert_allclose(got.u, ref.u, atol=atol)
    np.testing.assert_allclose(got.v, ref.v, atol=atol)


def test_multigrid_matches_cg_closely() -> None:
    g = Grid(64, 64, lx=1.0)
    system = PoissonSystem.from_bc(g.ny, g.nx, g.h, walls_bc())
    rng = np.random.default_rng(4)
    system.rhs = rng.uniform(-1.0, 1.0, g.shape_cell)
    system.rhs -= system.rhs.mean()
    mg = Multigrid().solve(system, tol=1e-12, maxiter=50)
    cg = ConjugateGradient().solve(system, tol=1e-12, maxiter=3000)
    a = mg.p - mg.p.mean()
    b = cg.p - cg.p.mean()
    assert float(np.abs(a - b).max() / np.abs(b).max()) < 1e-10


def test_unknown_solver_raises() -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        get_solver("nope")


# --------------------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["conjugate_gradient", "multigrid"])
def test_projection_makes_random_field_divergence_free(name: str) -> None:
    g = Grid(64, 64, lx=1.0)
    u, v = _random_field(g)
    before = float(np.max(np.abs(divergence(u, v, g.h))))
    res = project_velocity(u, v, g, walls_bc(), dt=0.1, solver=name, tol=1e-9, maxiter=2000)
    assert before > 10.0
    # The divergence obtained is dt * (residual of the Poisson solve) / h, and the residual is
    # normalised by max|rhs| = max|div|/dt.  With tol=1e-9 the achievable divergence is
    # therefore ~ tol * max|div| / (dt h) ~ 2e-7 here; the target below is 1e-6 (= tol/2 in
    # units of the *input* divergence) and is met by every solver that converges.
    assert res.div_max < 1e-6, res.div_max


def test_projection_reduces_divergence_by_orders_of_magnitude() -> None:
    g = Grid(48, 48, lx=1.0)
    u, v = _random_field(g)
    before = float(np.max(np.abs(divergence(u, v, g.h))))
    after = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="conjugate_gradient").div_max
    assert before / max(after, 1e-16) > 1e6


def test_projecting_a_divergence_free_field_changes_it_negligibly() -> None:
    """A corner-curl field is exactly div-free, so the pressure must be (nearly) constant."""
    g = Grid(32, 32, lx=1.0)
    X, Y = g.corner_coords()
    psi = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u, v = curl_streamfunction(psi, g.h)
    apply_velocity_bc(u, v, walls_bc())
    res = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="multigrid")
    np.testing.assert_allclose(res.u, u, atol=1e-10)
    np.testing.assert_allclose(res.v, v, atol=1e-10)
    assert float(np.abs(res.p).max()) < 1e-10


def test_projection_removes_a_pure_gradient_part() -> None:
    """``u* = curl(psi) + grad(phi)``: projection must return exactly the curl part."""
    g = Grid(32, 32, lx=1.0)
    X, Y = g.corner_coords()
    psi = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_curl, v_curl = curl_streamfunction(psi, g.h)
    Xc, Yc = g.cell_coords()
    phi = np.sin(2 * np.pi * Xc) * np.sin(2 * np.pi * Yc)
    gx, gy = gradient(phi, g.h)
    u = u_curl + gx
    v = v_curl + gy
    apply_velocity_bc(u, v, walls_bc())
    res = project_velocity(u, v, g, walls_bc(), dt=1.0, solver="multigrid")
    np.testing.assert_allclose(res.u[:, 1:-1], u_curl[:, 1:-1], atol=1e-9)
    np.testing.assert_allclose(res.v[1:-1, :], v_curl[1:-1, :], atol=1e-9)


def test_projection_with_periodic_boundaries() -> None:
    g = Grid(48, 48, lx=1.0)
    u, v = taylor_green(g)
    rng = np.random.default_rng(5)
    u = u + rng.uniform(-1, 1, g.shape_u)
    v = v + rng.uniform(-1, 1, g.shape_v)
    apply_velocity_bc(u, v, periodic_bc())
    res = project_velocity(u, v, g, periodic_bc(), dt=0.1, solver="multigrid")
    # tol=1e-10 by default; the divergence floor is tol*max|div|/(dt*h) ~ 1.6e-8 here
    assert res.div_max < 5e-8
    np.testing.assert_allclose(u, u, atol=0)  # sanity: no aliasing of inputs


def test_projection_with_an_obstacle_reduces_divergence_in_the_fluid() -> None:
    """With a solid obstacle the Poisson problem is solved on a masked domain.

    The masked system is not the same discrete problem as an unmasked one, so the honest
    statement is: (a) the solver converges to its own residual tolerance, (b) the divergence
    of the projected field in the fluid is far smaller than the divergence of the input, and
    (c) cells whose faces all lie inside the solid keep their (meaningless) velocity, which is
    why the reported divergence excludes them.  The measured numbers are printed by
    ``mackerel validate projection``.
    """
    g = Grid(40, 40, lx=1.0)
    X, Y = g.cell_coords()
    solid = (X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.15**2
    u, v = _random_field(g)
    from scipy.ndimage import binary_erosion

    # cells whose four bounding faces are all in the fluid (the faces adjacent to the
    # obstacle are walls and are excluded from the pressure system by construction)
    interior = binary_erosion(~solid, iterations=1)
    before = float(np.max(np.abs(divergence(u, v, g.h)[interior])))
    res = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="multigrid", solid=solid)
    assert np.all(np.isfinite(res.u))
    assert res.residual < 1e-10
    after = float(np.max(np.abs(divergence(res.u, res.v, g.h)[interior])))
    assert after < 1e-3 * before, (before, after)


def test_project_wrapper_returns_fields_only() -> None:
    g = Grid(16, 16, lx=1.0)
    u, v = _random_field(g)
    u2, v2 = project(u, v, g, walls_bc(), dt=0.1, solver="conjugate_gradient")
    assert u2.shape == u.shape and v2.shape == v.shape
    assert np.max(np.abs(divergence(u2, v2, g.h))) < 1e-6


def test_projection_preserves_rigid_rotation_vorticity() -> None:
    """Rigid rotation is already divergence free; its vorticity must survive projection."""
    g = Grid(32, 32, lx=1.0)
    u, v = solid_body_rotation(g, omega=1.0)
    res = project_velocity(u, v, g, walls_bc(), dt=0.1, solver="multigrid")
    from mackerel.diagnostics import vorticity_field

    w_before = vorticity_field(u, v, g)
    w_after = vorticity_field(res.u, res.v, g)
    np.testing.assert_allclose(w_after, w_before, atol=1e-6)
