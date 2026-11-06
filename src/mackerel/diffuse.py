"""Viscosity: implicit diffusion of velocity and scalars.

Why implicit
------------
Explicit diffusion needs ``dt < h^2 / (4 nu)``: on a 256² grid with ``nu = 1e-2`` that is
``dt < 4e-4``, and the limit tightens as the grid refines.  So the viscous step is backward
Euler,

    (u_new - u_old) / dt = nu * L u_new      <=>      (I - dt nu L) u_new = u_old

with ``L`` the 5-point Laplacian and ``alpha = nu dt / h^2``.  ``I - dt nu L`` has eigenvalues
``1 + 4 alpha (1 - cos)`` > 0, so the step is unconditionally stable for every ``dt``.  The same
operator applies to scalars (temperature, dye).  Each component is solved with a
diagonal-preconditioned conjugate gradient: no matrix assembly, and it converges in a handful
of iterations because ``alpha`` is small.

Where the unknowns live
-----------------------
``u`` sits on vertical faces: ``x = 0 .. lx`` (``nx + 1`` columns) and ``y = h/2 .. ly - h/2``
(``ny`` rows).  ``v`` is transposed: ``x = h/2 .. lx - h/2``, ``y = 0 .. ly``.  For each
component one direction is *normal* to two boundaries (the unknown lives on them) and the other
is *tangential* (the unknown sits half a cell inside).

* **Normal direction** -- the edge columns/rows are prescribed by the boundary conditions.  They
  are *data*, eliminated from the system: they do not appear as unknowns, but their value feeds
  the row just inside them through the right-hand side.  Eliminating rather than pinning them is
  what keeps the reduced matrix symmetric positive definite.
* **Tangential direction** -- no-slip puts the wall speed half a cell outside the first row.
  Substituting ``u_ghost = 2 u_wall - u_0`` into the 5-point Laplacian turns the outside
  neighbour into ``-u_0 + 2 u_wall``: net ``+alpha`` on the diagonal and ``2 alpha u_wall`` on
  the right-hand side.  Free-slip and outflow are zero-gradient, so the outside neighbour is
  simply absent (``alpha`` disappears from the diagonal and nothing is added).
* **Periodic direction** -- couplings wrap.  The duplicated seam entry that MAC storage keeps
  for the normal component (``u``'s last column, ``v``'s last row) is dropped before the solve
  and restored afterwards; keeping both copies makes the system singular along the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mackerel.grid import Array, BoundaryConditions, Grid, Kind, Side, apply_velocity_bc

__all__ = ["DiffusionResult", "diffuse", "diffuse_scalar"]

# Per-side coupling mode.
PERIODIC = "periodic"   # wraps to the opposite side
PRESCRIBED = "prescribed"  # the edge entry is data (normal-direction wall); eliminated
NEUMANN = "neumann"     # zero-gradient: the outside neighbour does not exist
GHOST = "ghost"         # Dirichlet half a cell outside: +alpha on the diagonal, 2 alpha w on RHS


@dataclass
class _System:
    """``A d = r`` on the interior unknowns, with boundary values eliminated.

    ``x_mode``/``y_mode`` are convenient defaults for tests and uniform boundaries.  The
    optional per-side modes allow a channel to have, for example, prescribed inflow on the
    left and zero-gradient outflow on the right without treating both as walls.
    """

    alpha: float
    x_mode: str
    y_mode: str
    fixed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    fixed_values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x_low: float = 0.0
    x_high: float = 0.0
    y_low: float = 0.0
    y_high: float = 0.0
    x_low_mode: str | None = None
    x_high_mode: str | None = None
    y_low_mode: str | None = None
    y_high_mode: str | None = None

    def _x_modes(self) -> tuple[str, str]:
        return self.x_low_mode or self.x_mode, self.x_high_mode or self.x_mode

    def _y_modes(self) -> tuple[str, str]:
        return self.y_low_mode or self.y_mode, self.y_high_mode or self.y_mode

    def apply(self, d: Array) -> Array:
        a = self.alpha
        ny, nx = d.shape
        if self.fixed.size == d.size:
            # Work on the reduced space: prescribed entries are not unknowns.
            d = np.where(self.fixed, 0.0, d)
        diag = np.full((ny, nx), 1.0 + 4.0 * a)
        nb = np.zeros_like(d)

        if self.x_mode == PERIODIC:
            nb[:, 0] += d[:, -1]
            nb[:, 1:] += d[:, :-1]
            nb[:, -1] += d[:, 0]
            nb[:, :-1] += d[:, 1:]
        else:
            nb[:, 1:] += d[:, :-1]
            nb[:, :-1] += d[:, 1:]
            xlo, xhi = self._x_modes()
            # Neumann: outside=inside, so remove one diagonal coupling.  No-slip ghost:
            # outside=2w-inside, giving a net +alpha on the diagonal.  A prescribed edge
            # is data and needs no diagonal adjustment on its (eliminated) row.
            if xlo == NEUMANN:
                diag[:, 0] -= a
            elif xlo == GHOST:
                diag[:, 0] += a
            if xhi == NEUMANN:
                diag[:, -1] -= a
            elif xhi == GHOST:
                diag[:, -1] += a

        if self.y_mode == PERIODIC:
            nb[0, :] += d[-1, :]
            nb[1:, :] += d[:-1, :]
            nb[-1, :] += d[0, :]
            nb[:-1, :] += d[1:, :]
        else:
            nb[1:, :] += d[:-1, :]
            nb[:-1, :] += d[1:, :]
            ylo, yhi = self._y_modes()
            if ylo == NEUMANN:
                diag[0, :] -= a
            elif ylo == GHOST:
                diag[0, :] += a
            if yhi == NEUMANN:
                diag[-1, :] -= a
            elif yhi == GHOST:
                diag[-1, :] += a

        out = diag * d - a * nb
        if self.fixed.size == d.size:
            out = np.where(self.fixed, 0.0, out)
        return out

    def source(self, shape: tuple[int, int]) -> Array:
        """Known data on the right-hand side: ghost walls and eliminated edge values."""
        a = self.alpha
        s = np.zeros(shape, dtype=np.float64)
        xlo, xhi = self._x_modes()
        ylo, yhi = self._y_modes()
        if xlo == GHOST:
            s[:, 0] += 2.0 * a * self.x_low
        if xhi == GHOST:
            s[:, -1] += 2.0 * a * self.x_high
        if ylo == GHOST:
            s[0, :] += 2.0 * a * self.y_low
        if yhi == GHOST:
            s[-1, :] += 2.0 * a * self.y_high
        if self.fixed.size and self.fixed_values.size == int(np.prod(shape)):
            values = self.fixed_values
            if xlo == PRESCRIBED:
                s[:, 1] += a * values[:, 0]
            if xhi == PRESCRIBED:
                s[:, -2] += a * values[:, -1]
            if ylo == PRESCRIBED:
                s[1, :] += a * values[0, :]
            if yhi == PRESCRIBED:
                s[-2, :] += a * values[-1, :]
        return s


@dataclass
class DiffusionResult:
    """Velocity after the implicit viscous step."""

    u: Array
    v: Array
    iterations: int
    residual: float


def _pcg(sys: _System, b: Array, x0: Array, tol: float, maxiter: int) -> tuple[Array, int, float]:
    """Diagonal-preconditioned CG for ``A d = b`` with ``d = 0`` on prescribed entries."""
    diag = np.maximum(sys.apply(np.ones_like(b)), 1e-12)
    pinned = sys.fixed.size == b.size
    if pinned:
        b = np.where(sys.fixed, 0.0, b)
        x0 = np.where(sys.fixed, 0.0, x0)

    def project(v: Array) -> Array:
        return np.where(sys.fixed, 0.0, v) if pinned else v

    x = project(x0.copy())
    r = project(b - sys.apply(x))
    z = r / diag
    d = z.copy()
    rz = float(np.sum(r * z))
    scale = max(float(np.max(np.abs(b))), 1e-30)
    res = float(np.max(np.abs(r))) / scale
    it = 0
    while it < maxiter and res > tol:
        ad = sys.apply(d)
        denom = float(np.sum(d * ad))
        if denom <= 0.0:
            break
        step = rz / denom
        x = project(x + step * d)
        r = project(r - step * ad)
        z = r / diag
        rz_new = float(np.sum(r * z))
        d = project(z + (rz_new / rz) * d)
        rz = rz_new
        res = float(np.max(np.abs(r))) / scale
        it += 1
    return x, it, res


def _velocity_system(
    component: str, shape: tuple[int, int], bc: BoundaryConditions, alpha: float
) -> _System:
    """Per-component system with independent low/high side conditions."""
    if component == "u":
        normal_sides = (Side.LEFT, Side.RIGHT)
        tangential_sides = (Side.BOTTOM, Side.TOP)
        normal_periodic = bc.is_x_periodic
        tangential_periodic = bc.is_y_periodic
        index = 0
    else:
        normal_sides = (Side.BOTTOM, Side.TOP)
        tangential_sides = (Side.LEFT, Side.RIGHT)
        normal_periodic = bc.is_y_periodic
        tangential_periodic = bc.is_x_periodic
        index = 1

    tangential_modes: list[str] = []
    ghost: list[float] = []
    for side in tangential_sides:
        sbc = bc.side(side)
        if tangential_periodic:
            tangential_modes.append(PERIODIC)
            ghost.append(0.0)
        elif sbc.kind in (Kind.NO_SLIP, Kind.INFLOW):
            tangential_modes.append(GHOST)
            ghost.append(sbc.velocity[index])
        else:  # FREE_SLIP and OUTFLOW have zero normal gradient
            tangential_modes.append(NEUMANN)
            ghost.append(0.0)

    fixed = np.zeros((0, 0), dtype=bool)
    values = np.zeros((0, 0), dtype=np.float64)
    normal_modes: list[str] = []
    if normal_periodic:
        normal_modes = [PERIODIC, PERIODIC]
    else:
        fixed = np.zeros(shape, dtype=bool)
        values = np.zeros(shape, dtype=np.float64)
        for side in normal_sides:
            sbc = bc.side(side)
            if sbc.kind is Kind.OUTFLOW:
                normal_modes.append(NEUMANN)
            else:
                # NO_SLIP/FREE_SLIP prescribe zero normal velocity; INFLOW prescribes its
                # signed inflow value.  Only OUTFLOW is an unknown zero-gradient face.
                normal_modes.append(PRESCRIBED)
        if component == "u":
            if normal_modes[0] == PRESCRIBED:
                fixed[:, 0] = True
                values[:, 0] = bc.left.velocity[0]
            if normal_modes[1] == PRESCRIBED:
                fixed[:, -1] = True
                values[:, -1] = bc.right.velocity[0]
        else:
            if normal_modes[0] == PRESCRIBED:
                fixed[0, :] = True
                values[0, :] = bc.bottom.velocity[1]
            if normal_modes[1] == PRESCRIBED:
                fixed[-1, :] = True
                values[-1, :] = bc.top.velocity[1]

    if component == "u":
        return _System(
            alpha=alpha, x_mode=normal_modes[0], y_mode=tangential_modes[0],
            fixed=fixed, fixed_values=values,
            y_low=ghost[0], y_high=ghost[1],
            x_low_mode=normal_modes[0], x_high_mode=normal_modes[1],
            y_low_mode=tangential_modes[0], y_high_mode=tangential_modes[1],
        )
    return _System(
        alpha=alpha, x_mode=tangential_modes[0], y_mode=normal_modes[0],
        fixed=fixed, fixed_values=values,
        x_low=ghost[0], x_high=ghost[1],
        x_low_mode=tangential_modes[0], x_high_mode=tangential_modes[1],
        y_low_mode=normal_modes[0], y_high_mode=normal_modes[1],
    )


def _drop_seam(component: str, field: Array, bc: BoundaryConditions) -> tuple[Array, int]:
    """Remove the duplicated seam entry of a periodic *normal* direction; return (field, axis)."""
    if component == "u" and bc.is_x_periodic:
        return np.delete(field, -1, axis=1), 1
    if component == "v" and bc.is_y_periodic:
        return np.delete(field, -1, axis=0), 0
    return field, -1


def diffuse(
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    nu: float,
    dt: float,
    tol: float = 1e-10,
    maxiter: int = 400,
) -> DiffusionResult:
    """Implicitly diffuse both velocity components with backward Euler."""
    bc.validate()
    if nu <= 0.0 or dt <= 0.0:
        return DiffusionResult(u=u.copy(), v=v.copy(), iterations=0, residual=0.0)

    alpha = nu * dt / (grid.h * grid.h)
    out: dict[str, Array] = {}
    iters = 0
    res = 0.0
    for component, velocity in (("u", u), ("v", v)):
        reduced, seam_axis = _drop_seam(component, velocity, bc)
        sys = _velocity_system(component, reduced.shape, bc, alpha)
        # The old field is both the initial guess and the right-hand side (implicit Euler keeps
        # the whole time level on the right); ``source`` adds the wall data.
        b = reduced + sys.source(reduced.shape)
        d, it, r = _pcg(sys, b, np.zeros_like(b), tol, maxiter)
        iters = max(iters, it)
        res = max(res, r)
        x = np.where(sys.fixed, sys.fixed_values, d) if sys.fixed.size else d
        if seam_axis >= 0:
            x = np.concatenate([x, np.take(x, [0], axis=seam_axis)], axis=seam_axis)
        out[component] = x

    out_u, out_v = out["u"], out["v"]
    apply_velocity_bc(out_u, out_v, bc)
    return DiffusionResult(u=out_u, v=out_v, iterations=iters, residual=res)


def diffuse_scalar(
    field: Array,
    grid: Grid,
    bc: BoundaryConditions,
    nu: float,
    dt: float,
    tol: float = 1e-10,
    maxiter: int = 400,
) -> Array:
    """Implicitly diffuse a cell-centred scalar (zero-gradient or periodic walls)."""
    if nu <= 0.0 or dt <= 0.0:
        return field.copy()
    alpha = nu * dt / (grid.h * grid.h)
    sys = _System(
        alpha=alpha,
        x_mode=PERIODIC if bc.is_x_periodic else NEUMANN,
        y_mode=PERIODIC if bc.is_y_periodic else NEUMANN,
    )
    out, _it, _res = _pcg(sys, field.copy(), np.zeros_like(field), tol, maxiter)
    return out
