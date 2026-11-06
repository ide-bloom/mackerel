"""Pressure projection: build the Poisson system, solve it, subtract the gradient.

The projection step
-------------------
After advection and diffusion the velocity ``u*`` generally has ``div u* != 0``.  The
pressure is the Lagrange multiplier that removes exactly that violation::

    u_new = u* - dt * grad(p),     div(u_new) = 0

Substituting gives the pressure Poisson equation::

    laplacian(p) = div(u*) / dt

Why the discrete operators fit together so well on a MAC grid: ``div`` maps faces to
centres, ``grad`` maps centres to faces, and on the staggered grid *they are exact negative
transposes of each other* (up to a sign and ``h``).  Therefore ``laplacian = div(grad)`` is
symmetric -- the property every solver in :mod:`mackerel.solvers` relies on.

Boundary conditions
-------------------
* **Solid wall / obstacle**: the normal velocity is prescribed, so the pressure must not
  change it: ``dp/dn = 0`` (Neumann).  Discretely this means the wall face pressure gradient
  is zero, i.e. the pressure of the wall neighbour is *mirrored* across the face.  When the
  wall is the domain boundary the mirror is implied by dropping the neighbour term; when the
  "wall" is a solid cell *inside* the domain (an obstacle) the mirror is explicit -- the
  solid cell's pressure is set equal to the fluid cell's while the gradient is evaluated.
* **Outflow**: the pressure is fixed (Dirichlet, value 0).  This is the only condition that
  makes the system non-singular; it also fixes the absolute level of ``p``.
* **Periodic**: the pressure wraps.

The pure-Neumann null space
---------------------------
With no outflow (all walls and/or periodic) the matrix has a one-dimensional null space: the
constant pressure vector.  The right-hand side is only solvable if its mean is zero, and the
solution is only defined up to a constant.  We handle this explicitly:

* ``PoissonSystem.null_space`` is true exactly when no side is an outflow;
* the right-hand side has its (area-weighted) mean removed before the solve;
* every solver projects the constant mode out of its iterates, i.e. converges to the
  minimum-norm solution;
* ``fixed_pressure`` can pin one cell instead, which is used by one of the tests to show the
  two approaches agree to a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mackerel.grid import (
    Array,
    BoolArray,
    BoundaryConditions,
    Grid,
    Kind,
    apply_velocity_bc,
    divergence,
    gradient,
)
from mackerel.obstacles import solid_face_masks
from mackerel.solvers import SolveResult, get_solver, laplacian_apply

__all__ = [
    "PoissonSystem",
    "ProjectionResult",
    "open_boundary_mask",
    "pressure_poisson_rhs",
    "project",
    "project_velocity",
]


# --------------------------------------------------------------------------------------
# the system
# --------------------------------------------------------------------------------------
@dataclass
class PoissonSystem:
    """A discretised ``laplacian(p) = rhs`` with its boundary structure precomputed.

    The operator is stored as two arrays of *face* couplings, which is exactly how the
    discrete Laplacian is built from the MAC faces:

    * ``cx[j, i]`` couples cell ``(j, i-1)`` and cell ``(j, i)`` (the ``u`` face between
      them), for ``i = 0 .. nx-1``; ``i = 0`` is the domain's left face.
    * ``cy[j, i]`` couples cell ``(j-1, i)`` and cell ``(j, i)`` (the ``v`` face), for
      ``j = 0 .. ny-1``; ``j = 0`` is the domain's bottom face.

    A coupling is 1 when the two cells exchange flow, 0 when the face is a wall (Neumann: the
    neighbour pressure is mirrored, which in the operator means no coupling at all) and 0 when
    the face is an open boundary -- in that case the outside pressure is the fixed value 0,
    so the *diagonal* of the adjacent cell gains 1 instead (Dirichlet).

    ``diag`` is the diagonal of the SPD operator ``-laplacian`` before dividing by ``h^2``.
    """

    h: float
    bc: BoundaryConditions
    solid: BoolArray                      # (ny, nx) True where the cell is not fluid
    rhs: Array                            # (ny, nx)
    guess: Array                          # (ny, nx)
    cx: Array                             # (ny, nx)  x-face couplings
    cy: Array                             # (ny, nx)  y-face couplings
    diag: Array                           # (ny, nx)
    null_space: bool
    fixed_pressure: bool = False
    # In free-surface mode ``solid`` means air: liquid-air faces have p=0 outside rather
    # than the zero-normal-gradient condition used for an obstacle.
    free_surface: bool = False

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_bc(
        cls,
        ny: int,
        nx: int,
        h: float,
        bc: BoundaryConditions,
        solid: BoolArray | None = None,
        null_space: bool | None = None,
        fixed_pressure: bool = False,
        free_surface: bool = False,
    ) -> PoissonSystem:
        """Build the operator structure for a grid with the given boundary conditions."""
        bc.validate()
        if solid is None:
            solid = np.zeros((ny, nx), dtype=bool)

        # --- interior couplings: a face between two fluid cells couples them
        # ``cx[:, 0]`` / ``cy[0, :]`` are *couplings to the opposite edge*, i.e. a periodic
        # wrap; they are zero (a wall) unless the corresponding direction is periodic.
        cx = np.ones((ny, nx), dtype=np.float64)
        cx[:, 1:] = np.where(solid[:, :-1] | solid[:, 1:], 0.0, 1.0)
        cx[:, 0] = 0.0
        cy = np.ones((ny, nx), dtype=np.float64)
        cy[1:, :] = np.where(solid[:-1, :] | solid[1:, :], 0.0, 1.0)
        cy[0, :] = 0.0

        diag = np.zeros((ny, nx), dtype=np.float64)
        # count each cell's coupled faces
        # x: cell i has faces i and i+1; face i couples (i-1, i) -> cell i uses cx[:, i+1]
        diag[:, :-1] += cx[:, 1:]
        diag[:, 1:] += cx[:, 1:]
        diag[:-1, :] += cy[1:, :]
        diag[1:, :] += cy[1:, :]

        if free_surface:
            # A liquid-air face sees atmospheric pressure p=0 on the air side.  It therefore
            # contributes a Dirichlet diagonal term to the liquid cell, unlike a solid wall
            # (whose coupling simply disappears).  Air rows are overwritten with identity
            # below; the XOR ensures only mixed faces contribute here.
            air_x = solid[:, :-1] ^ solid[:, 1:]
            air_y = solid[:-1, :] ^ solid[1:, :]
            diag[:, :-1] += air_x
            diag[:, 1:] += air_x
            diag[:-1, :] += air_y
            diag[1:, :] += air_y

        # --- the four domain boundaries.  Note which faces are actually *stored*: the
        # left face of the domain is cx[:, 0] (it separates cell 0 from the outside or, when
        # periodic, from cell nx-1); the right domain face is not an array entry at all.
        # The same holds for cy[0, :] and the top face.
        left, right = bc.left, bc.right
        bottom, top = bc.bottom, bc.top
        if bc.is_x_periodic:
            cx[:, 0] = np.where(solid[:, 0] | solid[:, -1], 0.0, 1.0)
            diag[:, 0] += cx[:, 0]
            diag[:, -1] += cx[:, 0]
        elif left.kind is Kind.OUTFLOW:
            diag[:, 0] += 1.0
        if right.kind is Kind.OUTFLOW:
            diag[:, -1] += 1.0
        if bc.is_y_periodic:
            cy[0, :] = np.where(solid[0, :] | solid[-1, :], 0.0, 1.0)
            diag[0, :] += cy[0, :]
            diag[-1, :] += cy[0, :]
        elif bottom.kind is Kind.OUTFLOW:
            diag[0, :] += 1.0
        if top.kind is Kind.OUTFLOW:
            diag[-1, :] += 1.0

        # solid cells are not unknowns: give them a harmless, invertible diagonal
        diag[solid] = 1.0
        cx[solid] = 0.0
        cy[solid] = 0.0

        if null_space is None:
            # Atmospheric air fixes the pressure level even in a closed domain.
            null_space = not bc.has_open_boundary and not (free_surface and np.any(solid))
        return cls(
            h=float(h),
            bc=bc,
            solid=solid,
            rhs=np.zeros((ny, nx), dtype=np.float64),
            guess=np.zeros((ny, nx), dtype=np.float64),
            cx=cx,
            cy=cy,
            diag=diag,
            null_space=bool(null_space),
            fixed_pressure=fixed_pressure,
            free_surface=free_surface,
        )

    # -- properties ---------------------------------------------------------------------
    @property
    def open_mask(self) -> BoolArray:
        """True for fluid cells (the unknowns of the system)."""
        return np.asarray(~self.solid, dtype=bool)

    @property
    def weight(self) -> Array:
        """Weights used for the mean over the active cells (0 in solid cells)."""
        return self.open_mask.astype(np.float64)

    @property
    def size(self) -> int:
        return int(np.count_nonzero(self.open_mask))

    def copy(self) -> PoissonSystem:
        return PoissonSystem(
            h=self.h,
            bc=self.bc,
            solid=self.solid.copy(),
            rhs=self.rhs.copy(),
            guess=self.guess.copy(),
            cx=self.cx.copy(),
            cy=self.cy.copy(),
            diag=self.diag.copy(),
            null_space=self.null_space,
            fixed_pressure=self.fixed_pressure,
            free_surface=self.free_surface,
        )


# --------------------------------------------------------------------------------------
# right-hand side
# --------------------------------------------------------------------------------------
def open_boundary_mask(bc: BoundaryConditions, shape: tuple[int, int]) -> BoolArray:
    """Cells that touch an outflow (Dirichlet pressure) side."""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    if bc.left.kind is Kind.OUTFLOW:
        mask[:, 0] = True
    if bc.right.kind is Kind.OUTFLOW:
        mask[:, -1] = True
    if bc.bottom.kind is Kind.OUTFLOW:
        mask[0, :] = True
    if bc.top.kind is Kind.OUTFLOW:
        mask[-1, :] = True
    return mask


def pressure_poisson_rhs(u: Array, v: Array, grid: Grid, system: PoissonSystem,
                         dt: float) -> Array:
    """Right-hand side of the SPD system ``(-laplacian) p = -div(u*) / dt``.

    Derivation, with signs spelled out because they are the easiest thing to get wrong:
    ``u_new = u* - dt grad(p)`` and ``div(u_new) = 0`` give ``laplacian(p) = div(u*)/dt``.
    The Laplacian is negative semi-definite, so the solvers solve ``(-laplacian) p = -div/dt``
    with ``-laplacian`` positive definite (that is what :func:`laplacian_apply` applies).
    Getting this sign wrong does not crash anything -- it *doubles* the divergence instead of
    removing it, which is exactly how it was caught in the test suite.

    In solid cells the right-hand side is set to zero (they are not unknowns) and, when the
    system is singular, the area-weighted mean is removed so the equation is solvable.
    """
    div = -divergence(u, v, grid.h) / dt
    div = np.where(system.open_mask, div, 0.0)
    if system.null_space and not system.fixed_pressure:
        w = system.weight
        total = float(np.sum(w))
        if total > 0.0:
            div = div - float(np.sum(div * w)) / total
    return div


# --------------------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------------------
@dataclass
class ProjectionResult:
    """Velocity after projection together with everything the caller may want to know."""

    u: Array
    v: Array
    p: Array
    iterations: int
    residual: float
    div_max: float
    div_l2: float
    solve: SolveResult | None = field(default=None, repr=False)


def _divergence_stats(
    u: Array, v: Array, grid: Grid, fluid: BoolArray | None = None
) -> tuple[float, float]:
    """``(max |div|, L2 div)`` over the fluid cells.

    Cells inside a solid have no physical divergence: the velocity there is whatever the
    obstacle's no-slip data says, and the discrete divergence of that field is meaningless.
    Measuring it and calling the result "the projection failed" would be a category error, so
    the statistics exclude solid cells -- and the tests assert this explicitly.
    """
    d = divergence(u, v, grid.h)
    if fluid is not None:
        d = np.where(fluid, d, 0.0)
    return float(np.max(np.abs(d))), float(np.sqrt(np.sum(d * d) * grid.cell_area))


def project_velocity(
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    system: PoissonSystem | None = None,
    solver: str = "conjugate_gradient",
    tol: float = 1e-10,
    maxiter: int = 500,
    solid: BoolArray | None = None,
    keep_solid_velocity: bool = True,
    free_surface: bool = False,
) -> ProjectionResult:
    """Make ``(u, v)`` discretely divergence free and return the pressure that did it.

    The steps are exactly the four lines of the maths: build ``rhs = div(u)/dt``, solve
    ``laplacian(p) = rhs``, subtract ``dt * grad(p)`` from the interior faces, and restore the
    boundary conditions (the boundary normal velocities are prescribed, so the pressure
    gradient must not be applied there).
    """
    if system is None:
        solid_mask = np.zeros(grid.shape_cell, dtype=bool) if solid is None else solid
        system = PoissonSystem.from_bc(
            grid.ny, grid.nx, grid.h, bc, solid=solid_mask, free_surface=free_surface
        )
    system.rhs = pressure_poisson_rhs(u, v, grid, system, dt)
    system.guess = np.zeros_like(system.rhs)
    result = get_solver(solver).solve(system, tol=tol, maxiter=maxiter)
    p = np.where(system.open_mask, result.p, 0.0)
    if system.null_space:
        w = system.weight
        total = float(np.sum(w))
        if total > 0.0:
            p = p - float(np.sum(p * w)) / total

    u_new = u.copy()
    v_new = v.copy()
    gx, gy = gradient(p, grid.h, bc.is_x_periodic, bc.is_y_periodic)
    # OUTFLOW is a pressure Dirichlet boundary (p=0 outside), not a Neumann wall.  The
    # generic gradient helper quite correctly returns zero for an ordinary wall, so add the
    # one-sided Dirichlet differences here before correcting the boundary normal faces.
    if bc.left.kind is Kind.OUTFLOW:
        gx[:, 0] = p[:, 0] / grid.h
    if bc.right.kind is Kind.OUTFLOW:
        gx[:, -1] = -p[:, -1] / grid.h
    if bc.bottom.kind is Kind.OUTFLOW:
        gy[0, :] = p[0, :] / grid.h
    if bc.top.kind is Kind.OUTFLOW:
        gy[-1, :] = -p[-1, :] / grid.h
    if system.free_surface:
        # Only faces touching liquid receive a pressure correction.  Air-air faces are
        # allowed to keep their extrapolated particle velocities; liquid-air faces see p=0
        # on the air side through the ordinary difference stencil.
        liquid = system.open_mask
        active_u = np.zeros_like(u_new, dtype=bool)
        active_v = np.zeros_like(v_new, dtype=bool)
        active_u[:, 1:-1] = liquid[:, :-1] | liquid[:, 1:]
        active_u[:, 0] = liquid[:, 0] | (liquid[:, -1] if bc.is_x_periodic else False)
        active_u[:, -1] = liquid[:, -1] | (liquid[:, 0] if bc.is_x_periodic else False)
        active_v[1:-1, :] = liquid[:-1, :] | liquid[1:, :]
        active_v[0, :] = liquid[0, :] | (liquid[-1, :] if bc.is_y_periodic else False)
        active_v[-1, :] = liquid[-1, :] | (liquid[0, :] if bc.is_y_periodic else False)
        u_new[active_u] -= dt * gx[active_u]
        v_new[active_v] -= dt * gy[active_v]
    else:
        u_new[:, 1:-1] -= dt * gx[:, 1:-1]
        v_new[1:-1, :] -= dt * gy[1:-1, :]
    if bc.is_x_periodic:
        # the faces at x = 0 and x = lx are the same face, stored twice; the gradient must be
        # subtracted on both so the two copies stay identical and the divergence of the
        # boundary cell sees the corrected face (column 0 is reset from column -1 below)
        if system.free_surface:
            u_new[:, -1] -= dt * gx[:, -1] * active_u[:, -1]
            u_new[:, 0] -= dt * gx[:, 0] * active_u[:, 0]
        else:
            u_new[:, -1] -= dt * gx[:, -1]
            u_new[:, 0] -= dt * gx[:, 0]
    if bc.is_y_periodic:
        if system.free_surface:
            v_new[-1, :] -= dt * gy[-1, :] * active_v[-1, :]
            v_new[0, :] -= dt * gy[0, :] * active_v[0, :]
        else:
            v_new[-1, :] -= dt * gy[-1, :]
            v_new[0, :] -= dt * gy[0, :]
    if not bc.is_x_periodic:
        if system.free_surface:
            u_new[:, 0] -= dt * gx[:, 0] * active_u[:, 0]
            u_new[:, -1] -= dt * gx[:, -1] * active_u[:, -1]
        else:
            u_new[:, 0] -= dt * gx[:, 0]
            u_new[:, -1] -= dt * gx[:, -1]
    if not bc.is_y_periodic:
        if system.free_surface:
            v_new[0, :] -= dt * gy[0, :] * active_v[0, :]
            v_new[-1, :] -= dt * gy[-1, :] * active_v[-1, :]
        else:
            v_new[0, :] -= dt * gy[0, :]
            v_new[-1, :] -= dt * gy[-1, :]
    if keep_solid_velocity:
        # A solid face is prescribed data just like a domain wall.  The masked Poisson
        # operator has no coupling across it, so the pressure gradient must not alter its
        # input value (zero for a fixed obstacle, nonzero for a moving one).
        if np.any(system.solid) and not system.free_surface:
            um, vm = solid_face_masks(system.solid)
            u_new[um] = u[um]
            v_new[vm] = v[vm]
        apply_velocity_bc(u_new, v_new, bc)
    dmax, dl2 = _divergence_stats(u_new, v_new, grid, system.open_mask)
    return ProjectionResult(
        u=u_new,
        v=v_new,
        p=p,
        iterations=result.iterations,
        residual=result.residual,
        div_max=dmax,
        div_l2=dl2,
        solve=result,
    )


def project(
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float = 1.0,
    **kwargs: object,
) -> tuple[Array, Array]:
    """Convenience wrapper returning just the projected velocity."""
    res = project_velocity(u, v, grid, bc, dt, **kwargs)  # type: ignore[arg-type]
    return res.u, res.v


def residual_norm(p: Array, system: PoissonSystem) -> float:
    """Max-norm of the Poisson residual for a given pressure (diagnostics/tests)."""
    return float(np.max(np.abs(system.rhs - laplacian_apply(p, system))))
