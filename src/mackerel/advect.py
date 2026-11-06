"""Advection: semi-Lagrangian, MacCormack and BFECC, all vectorised over the whole grid.

Three schemes, one interface.  ``advect_velocity`` moves the velocity field and
``advect_scalar`` moves a cell-centred scalar (dye, temperature) with the same velocity
field; both take the immutable :class:`~mackerel.grid.BoundaryConditions`.

The idea behind all of them
---------------------------
Advection says a quantity is carried by the flow, ``dφ/dt = 0`` along a trajectory.  The
semi-Lagrangian method solves that by geometry: for every destination point ``x`` integrate
the velocity field *backwards* over one step to find where the fluid came from, then copy
the value from there.  Nothing is ever "unstable", because we never evaluate a spatial
derivative of ``φ`` -- but the interpolation used to copy the value smooths it, and that
smoothing is numerical diffusion.

MacCormack and BFECC both attack that smoothing by estimating the interpolation error and
subtracting it, then clamping the correction so that no new extrema appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mackerel.grid import (
    Array,
    BoundaryConditions,
    Grid,
    bilinear,
    pad_scalar,
    pad_velocity,
)

__all__ = [
    "AdvectionOptions",
    "Scheme",
    "advect_scalar",
    "advect_scalar_bfecc",
    "advect_scalar_maccormack",
    "advect_scalar_semi_lagrangian",
    "advect_velocity",
    "backtrace",
    "semi_lagrangian_u",
    "semi_lagrangian_v",
]


class Scheme(Enum):
    """Which advection scheme to use."""

    SEMI_LAGRANGIAN = "semi_lagrangian"
    MACCORMACK = "maccormack"
    BFECC = "bfecc"


@dataclass(frozen=True)
class AdvectionOptions:
    """Options shared by the advection routines.

    ``order`` selects the RK integrator used for the backtrace: 2 = midpoint rule, 3 = RK3.
    ``limiter`` clamps the MacCormack/BFECC correction to the range of the stencil it is
    correcting (a local min/max limiter); turn it off only to *demonstrate* the overshoot.
    """

    scheme: Scheme = Scheme.SEMI_LAGRANGIAN
    limiter: bool = True
    order: int = 2


# --------------------------------------------------------------------------------------
# backtrace
# --------------------------------------------------------------------------------------
def backtrace(
    up: Array,
    vp: Array,
    grid: Grid,
    x: Array,
    y: Array,
    dt: float,
    order: int = 2,
) -> tuple[Array, Array]:
    """Integrate ``dx/dt = u(x)`` backwards by ``dt`` from every point in ``(x, y)``.

    ``up``/``vp`` are the *padded* velocity arrays from :func:`mackerel.grid.pad_velocity`,
    so the sampled offsets are ``x0=-h, y0=-h/2`` for ``u`` and ``x0=-h/2, y0=-h`` for
    ``v`` (see that function's docstring).  Sampling inside the ghost ring is what makes the
    trajectory slow to the wall speed instead of overshooting through it.
    """
    h = grid.h
    if order == 2:
        u1 = bilinear(up, x, y, -h, -0.5 * h, h)
        v1 = bilinear(vp, x, y, -0.5 * h, -h, h)
        xm = x - 0.5 * dt * u1
        ym = y - 0.5 * dt * v1
        u2 = bilinear(up, xm, ym, -h, -0.5 * h, h)
        v2 = bilinear(vp, xm, ym, -0.5 * h, -h, h)
        return x - dt * u2, y - dt * v2
    if order == 3:
        u1 = bilinear(up, x, y, -h, -0.5 * h, h)
        v1 = bilinear(vp, x, y, -0.5 * h, -h, h)
        x1, y1 = x - 0.5 * dt * u1, y - 0.5 * dt * v1
        u2 = bilinear(up, x1, y1, -h, -0.5 * h, h)
        v2 = bilinear(vp, x1, y1, -0.5 * h, -h, h)
        x2, y2 = x - 0.5 * dt * u2, y - 0.5 * dt * v2
        u3 = bilinear(up, x2, y2, -h, -0.5 * h, h)
        v3 = bilinear(vp, x2, y2, -0.5 * h, -h, h)
        x3, y3 = x - dt * u3, y - dt * v3
        u4 = bilinear(up, x3, y3, -h, -0.5 * h, h)
        v4 = bilinear(vp, x3, y3, -0.5 * h, -h, h)
        return (
            x - dt / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4),
            y - dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4),
        )
    raise ValueError(f"unsupported backtrace order {order}")


# --------------------------------------------------------------------------------------
# limiters
# --------------------------------------------------------------------------------------
def _local_bounds(
    values: Array, radius: int = 1, wrap_x: bool = False, wrap_y: bool = False
) -> tuple[Array, Array]:
    """Per-cell min/max over the ``(2*radius+1)^2`` neighbourhood.

    Periodicity matters here: padding an edge-clamped copy at a periodic seam makes the
    limiter clip a value that is perfectly admissible (it arrived from the far side of the
    domain), which drains mass out of the field one cell per step.  ``wrap_x``/``wrap_y``
    select periodic padding on the corresponding axis.
    """
    p = np.pad(values, radius, mode="wrap" if (wrap_x and wrap_y) else "edge")
    ny, nx = values.shape
    lo = np.full_like(values, np.inf)
    hi = np.full_like(values, -np.inf)
    for dj in range(2 * radius + 1):
        for di in range(2 * radius + 1):
            block = p[dj : dj + ny, di : di + nx]
            np.minimum(lo, block, out=lo)
            np.maximum(hi, block, out=hi)
    return lo, hi


def _clamp_to_bounds(corrected: Array, lo: Array, hi: Array) -> Array:
    return np.clip(corrected, lo, hi)


def _bounds_like(values: Array, bc: BoundaryConditions) -> tuple[Array, Array]:
    """Local 3x3 bounds with the periodic axes wrapped (see :func:`_local_bounds`)."""
    return _local_bounds(values, 1, bc.is_x_periodic, bc.is_y_periodic)


# --------------------------------------------------------------------------------------
# scalar advection
# --------------------------------------------------------------------------------------
def advect_scalar_semi_lagrangian(
    field: Array,
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    order: int = 2,
) -> Array:
    """Semi-Lagrangian advection of a cell-centred scalar (RK2 backtrace by default)."""
    cp = pad_scalar(field, bc)
    up, vp = pad_velocity(u, v, bc)
    X, Y = grid.cell_coords()
    xd, yd = backtrace(up, vp, grid, X, Y, dt, order=order)
    out = bilinear(cp, xd, yd, -0.5 * grid.h, -0.5 * grid.h, grid.h)
    return np.asarray(out, dtype=np.float64)


def advect_scalar_maccormack(
    field: Array,
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    order: int = 2,
    limiter: bool = True,
) -> Array:
    """MacCormack advection of a cell-centred scalar.

    Forward semi-Lagrangian sweep ``fwd = SL(phi)``, then a backward sweep from the departure
    points ``back = SL_back(fwd)``.  The difference ``(back - phi)/2`` estimates the
    interpolation error, so the corrected value is ``fwd - (back - phi)/2``.  Without the limiter this is
    third-order in smooth regions and produces new extrema at fronts; with the limiter the
    correction is clamped to the local stencil range.
    """
    cp = pad_scalar(field, bc)
    up, vp = pad_velocity(u, v, bc)
    h = grid.h
    X, Y = grid.cell_coords()

    xd, yd = backtrace(up, vp, grid, X, Y, dt, order=order)
    fwd = bilinear(cp, xd, yd, -0.5 * h, -0.5 * h, h)

    # Backward sweep: apply the *adjoint* semi-Lagrangian step to ``fwd`` on the grid, i.e.
    # trace forward in time (velocity reversed) from the grid points themselves, not from
    # the departure points.
    fwd_p = pad_scalar(fwd, bc)
    xb, yb = backtrace(up, vp, grid, X, Y, -dt, order=order)
    back = bilinear(fwd_p, xb, yb, -0.5 * h, -0.5 * h, h)

    corrected = fwd - 0.5 * (back - cp[1:-1, 1:-1])
    if limiter:
        corrected = _clamp_to_bounds(corrected, *_bounds_like(cp[1:-1, 1:-1], bc))
    return np.asarray(corrected, dtype=np.float64)


def advect_scalar_bfecc(
    field: Array,
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    order: int = 2,
    limiter: bool = True,
) -> Array:
    """BFECC (Back and Forth Error Compensation and Correction) advection of a scalar.

    Identical in spirit to MacCormack but the error estimate is formed symmetrically:
    advect forward, advect the result back, and compare with the original; then subtract
    half the discrepancy and advect forward once more.  Cost: three semi-Lagrangian sweeps
    (MacCormack needs two).
    """
    cp = pad_scalar(field, bc)
    up, vp = pad_velocity(u, v, bc)
    h = grid.h
    X, Y = grid.cell_coords()

    xd, yd = backtrace(up, vp, grid, X, Y, dt, order=order)
    fwd = bilinear(cp, xd, yd, -0.5 * h, -0.5 * h, h)
    fwd_p = pad_scalar(fwd, bc)
    xb, yb = backtrace(up, vp, grid, X, Y, -dt, order=order)
    back = bilinear(fwd_p, xb, yb, -0.5 * h, -0.5 * h, h)

    # the error estimate, half-subtracted
    field0 = cp[1:-1, 1:-1]
    lo, hi = _bounds_like(field0, bc)
    corrected_field = field0 - 0.5 * (back - field0)
    if limiter:
        corrected_field = _clamp_to_bounds(corrected_field, lo, hi)
    corr_p = pad_scalar(corrected_field, bc)
    out = bilinear(corr_p, xd, yd, -0.5 * h, -0.5 * h, h)
    if limiter:
        out = _clamp_to_bounds(out, *_bounds_like(corrected_field, bc))
    return np.asarray(out, dtype=np.float64)


def advect_scalar(
    field: Array,
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    options: AdvectionOptions | None = None,
) -> Array:
    """Advect a cell-centred scalar with the scheme named in ``options``."""
    opts = options or AdvectionOptions()
    match opts.scheme:
        case Scheme.SEMI_LAGRANGIAN:
            return advect_scalar_semi_lagrangian(field, u, v, grid, bc, dt, opts.order)
        case Scheme.MACCORMACK:
            return advect_scalar_maccormack(field, u, v, grid, bc, dt, opts.order, opts.limiter)
        case Scheme.BFECC:
            return advect_scalar_bfecc(field, u, v, grid, bc, dt, opts.order, opts.limiter)


# --------------------------------------------------------------------------------------
# velocity advection
# --------------------------------------------------------------------------------------
def _advect_faces(
    values: Array,
    up: Array,
    vp: Array,
    grid: Grid,
    dt: float,
    order: int,
    x0: float,
    y0: float,
    X: Array,
    Y: Array,
) -> Array:
    """One semi-Lagrangian sweep of a face-centred velocity component."""
    xd, yd = backtrace(up, vp, grid, X, Y, dt, order=order)
    return np.asarray(bilinear(values, xd, yd, x0, y0, grid.h), dtype=np.float64)


def semi_lagrangian_u(
    u: Array, v: Array, grid: Grid, bc: BoundaryConditions, dt: float, order: int = 2
) -> Array:
    """Semi-Lagrangian advection of the ``u`` component alone."""
    up, vp = pad_velocity(u, v, bc)
    X, Y = grid.u_coords()
    return _advect_faces(u, up, vp, grid, dt, order, -grid.h, -0.5 * grid.h, X, Y)


def semi_lagrangian_v(
    u: Array, v: Array, grid: Grid, bc: BoundaryConditions, dt: float, order: int = 2
) -> Array:
    """Semi-Lagrangian advection of the ``v`` component alone."""
    up, vp = pad_velocity(u, v, bc)
    X, Y = grid.v_coords()
    return _advect_faces(v, up, vp, grid, dt, order, -0.5 * grid.h, -grid.h, X, Y)


def advect_velocity(
    u: Array,
    v: Array,
    grid: Grid,
    bc: BoundaryConditions,
    dt: float,
    options: AdvectionOptions | None = None,
) -> tuple[Array, Array]:
    """Advect both velocity components with the same scheme.

    The velocity is advected by itself (non-linear term), so the two components must see the
    *same* padded velocity field and the same departure points; that is why both sweeps are
    driven here rather than in two independent calls.  MacCormack/BFECC are applied to each
    component with the same limiter, and the wall normal faces are restored afterwards by
    :func:`mackerel.grid.apply_velocity_bc` in the solver loop.
    """
    opts = options or AdvectionOptions()
    up, vp = pad_velocity(u, v, bc)
    h = grid.h
    Xu, Yu = grid.u_coords()
    Xv, Yv = grid.v_coords()
    xdu, ydu = backtrace(up, vp, grid, Xu, Yu, dt, opts.order)
    xdv, ydv = backtrace(up, vp, grid, Xv, Yv, dt, opts.order)

    u_new = np.asarray(bilinear(up, xdu, ydu, -h, -0.5 * h, h), dtype=np.float64)
    v_new = np.asarray(bilinear(vp, xdv, ydv, -0.5 * h, -h, h), dtype=np.float64)
    if opts.scheme is Scheme.SEMI_LAGRANGIAN:
        return u_new, v_new

    # --- backward sweep for the error estimate (traced forward in time from the grid points)
    up2, vp2 = pad_velocity(u_new, v_new, bc)
    uxb, uyb = backtrace(up2, vp2, grid, Xu, Yu, -dt, opts.order)
    vxb, vyb = backtrace(up2, vp2, grid, Xv, Yv, -dt, opts.order)
    u_back = np.asarray(bilinear(up2, uxb, uyb, -h, -0.5 * h, h), dtype=np.float64)
    v_back = np.asarray(bilinear(vp2, vxb, vyb, -0.5 * h, -h, h), dtype=np.float64)

    if opts.scheme is Scheme.BFECC:
        # BFECC corrects the *old* field before the final forward sweep:
        #   phi_c = phi^n + (phi^n - A^R A phi^n) / 2,   phi^{n+1} = A phi_c.
        # Starting from ``u_new`` here would reduce this to another MacCormack step.
        u_corrected = u + 0.5 * (u - u_back)
        v_corrected = v + 0.5 * (v - v_back)
        if opts.limiter:
            u_corrected = _clamp_to_bounds(
                u_corrected, *_local_bounds(u, 1, bc.is_x_periodic, bc.is_y_periodic)
            )
            v_corrected = _clamp_to_bounds(
                v_corrected, *_local_bounds(v, 1, bc.is_x_periodic, bc.is_y_periodic)
            )
        upc, vpc = pad_velocity(u_corrected, v_corrected, bc)
        u_corr = np.asarray(bilinear(upc, xdu, ydu, -h, -0.5 * h, h), dtype=np.float64)
        v_corr = np.asarray(bilinear(vpc, xdv, ydv, -h, -0.5 * h, h), dtype=np.float64)
        if opts.limiter:
            u_corr = _clamp_to_bounds(
                u_corr, *_local_bounds(u_corrected, 1, bc.is_x_periodic, bc.is_y_periodic)
            )
            v_corr = _clamp_to_bounds(
                v_corr, *_local_bounds(v_corrected, 1, bc.is_x_periodic, bc.is_y_periodic)
            )
        return u_corr, v_corr

    # MacCormack applies the error estimate directly to the forward sweep.
    u_corr = u_new - 0.5 * (u_back - u)
    v_corr = v_new - 0.5 * (v_back - v)
    if opts.limiter:
        u_corr = _clamp_to_bounds(
            u_corr, *_local_bounds(u, 1, bc.is_x_periodic, bc.is_y_periodic)
        )
        v_corr = _clamp_to_bounds(
            v_corr, *_local_bounds(v, 1, bc.is_x_periodic, bc.is_y_periodic)
        )
    return u_corr, v_corr
