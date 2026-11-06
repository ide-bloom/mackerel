"""Staggered MAC grid geometry, bilinear sampling and boundary conditions.

Coordinate conventions
----------------------
``Grid(nx, ny, lx)`` describes the rectangle ``[0, lx] x [0, ly]`` with square cells of
size ``h = lx / nx`` and ``ly = ny * h``.

* cell centres: ``x_i = (i + 1/2) h``, ``y_j = (j + 1/2) h``  -> shape ``(ny, nx)``
* ``u`` (x-velocity) lives on vertical faces, ``x = i h``, ``y = (j + 1/2) h``
  -> shape ``(ny, nx + 1)``
* ``v`` (y-velocity) lives on horizontal faces, ``x = (i + 1/2) h``, ``y = j h``
  -> shape ``(ny + 1, nx)``
* pressure, temperature, dye and other scalars live at cell centres -> ``(ny, nx)``
* vorticity lives at cell corners, ``x = i h``, ``y = j h`` -> ``(ny + 1, nx + 1)``

All arrays are indexed ``[j, i]`` = ``[y, x]``, are ``float64``, and are indexed in
"logical" order: ``i`` along ``u`` is the face index ``0 .. nx``, so no half-cell
bookkeeping leaks into array bounds -- only into the coordinate helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


# --------------------------------------------------------------------------------------
# grid geometry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Pure geometry of a MAC grid.  Holds no field data and no boundary information."""

    nx: int
    ny: int
    lx: float = 1.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.lx <= 0.0:
            raise ValueError("lx must be positive")

    @property
    def h(self) -> float:
        """Cell size (cells are square)."""
        return self.lx / self.nx

    @property
    def ly(self) -> float:
        """Domain height implied by square cells."""
        return self.ny * self.h

    @property
    def shape_cell(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def shape_u(self) -> tuple[int, int]:
        return (self.ny, self.nx + 1)

    @property
    def shape_v(self) -> tuple[int, int]:
        return (self.ny + 1, self.nx)

    @property
    def shape_corner(self) -> tuple[int, int]:
        return (self.ny + 1, self.nx + 1)

    @property
    def cell_area(self) -> float:
        return self.h * self.h

    def __str__(self) -> str:
        return f"Grid({self.nx}x{self.ny}, lx={self.lx:g}, h={self.h:g})"

    # -- coordinate axes ---------------------------------------------------------------
    def x_centers(self) -> Array:
        return (np.arange(self.nx, dtype=np.float64) + 0.5) * self.h

    def y_centers(self) -> Array:
        return (np.arange(self.ny, dtype=np.float64) + 0.5) * self.h

    def x_faces(self) -> Array:
        return np.arange(self.nx + 1, dtype=np.float64) * self.h

    def y_faces(self) -> Array:
        return np.arange(self.ny + 1, dtype=np.float64) * self.h

    # -- coordinate meshes (shapes match the fields) -----------------------------------
    def cell_coords(self) -> tuple[Array, Array]:
        """``(X, Y)`` at cell centres, shape ``(ny, nx)``."""
        return np.meshgrid(self.x_centers(), self.y_centers())

    def u_coords(self) -> tuple[Array, Array]:
        """``(X, Y)`` at ``u`` faces, shape ``(ny, nx+1)``."""
        return np.meshgrid(self.x_faces(), self.y_centers())

    def v_coords(self) -> tuple[Array, Array]:
        """``(X, Y)`` at ``v`` faces, shape ``(ny+1, nx)``."""
        return np.meshgrid(self.x_centers(), self.y_faces())

    def corner_coords(self) -> tuple[Array, Array]:
        """``(X, Y)`` at cell corners, shape ``(ny+1, nx+1)``."""
        return np.meshgrid(self.x_faces(), self.y_faces())

    # -- field factories ---------------------------------------------------------------
    def zeros_cell(self) -> Array:
        return np.zeros(self.shape_cell, dtype=np.float64)

    def zeros_u(self) -> Array:
        return np.zeros(self.shape_u, dtype=np.float64)

    def zeros_v(self) -> Array:
        return np.zeros(self.shape_v, dtype=np.float64)

    def zeros_corner(self) -> Array:
        return np.zeros(self.shape_corner, dtype=np.float64)

    def noise_u(self, seed: int = 0) -> Array:
        """Random ``u`` field in ``[-1, 1]`` (tests / demonstrations)."""
        return np.random.default_rng(seed).uniform(-1.0, 1.0, self.shape_u)

    def noise_v(self, seed: int = 1) -> Array:
        """Random ``v`` field in ``[-1, 1]``."""
        return np.random.default_rng(seed).uniform(-1.0, 1.0, self.shape_v)

    # -- interpolation to cell centres -------------------------------------------------
    def u_to_centers(self, u: Array) -> Array:
        """Average ``u`` faces onto cell centres: shape ``(ny, nx)``."""
        return 0.5 * (u[:, :-1] + u[:, 1:])

    def v_to_centers(self, v: Array) -> Array:
        """Average ``v`` faces onto cell centres: shape ``(ny, nx)``."""
        return 0.5 * (v[:-1, :] + v[1:, :])


# --------------------------------------------------------------------------------------
# bilinear interpolation
# --------------------------------------------------------------------------------------
def bilinear(
    values: Array,
    x: Array | float,
    y: Array | float,
    x0: float,
    y0: float,
    h: float,
    clamp: bool = True,
) -> Array:
    """Bilinearly sample ``values`` at the points ``(x, y)``.

    ``values[j, i]`` is assumed to live at ``(x0 + i*h, y0 + j*h)``.  This is the workhorse
    for semi-Lagrangian advection: pass a *padded* array and it will sample inside the
    ghost ring (see :func:`pad_velocity`).

    With ``clamp=True`` points outside the sampled array are clamped to the nearest edge
    sample; with ``clamp=False`` they are extrapolated linearly.
    """
    ny, nx = values.shape
    fx = (np.asarray(x, dtype=np.float64) - x0) / h
    fy = (np.asarray(y, dtype=np.float64) - y0) / h
    if clamp:
        fx = np.clip(fx, 0.0, nx - 1.0)
        fy = np.clip(fy, 0.0, ny - 1.0)
    i0 = np.clip(np.floor(fx).astype(np.intp), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(np.intp), 0, ny - 2)
    tx = fx - i0
    ty = fy - j0
    v00 = values[j0, i0]
    v10 = values[j0, i0 + 1]
    v01 = values[j0 + 1, i0]
    v11 = values[j0 + 1, i0 + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def sample_cell(field: Array, grid: Grid, x: Array, y: Array, clamp: bool = True) -> Array:
    """Bilinearly sample a cell-centred field at physical points ``(x, y)``."""
    return bilinear(field, x, y, 0.5 * grid.h, 0.5 * grid.h, grid.h, clamp=clamp)


def sample_u(u: Array, grid: Grid, x: Array, y: Array, clamp: bool = True) -> Array:
    """Bilinearly sample the ``u`` face field."""
    return bilinear(u, x, y, 0.0, 0.5 * grid.h, grid.h, clamp=clamp)


def sample_v(v: Array, grid: Grid, x: Array, y: Array, clamp: bool = True) -> Array:
    """Bilinearly sample the ``v`` face field."""
    return bilinear(v, x, y, 0.5 * grid.h, 0.0, grid.h, clamp=clamp)


# --------------------------------------------------------------------------------------
# boundary conditions
# --------------------------------------------------------------------------------------
class Kind(Enum):
    """Kind of boundary condition on one side of the domain."""

    NO_SLIP = "no_slip"
    FREE_SLIP = "free_slip"
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    PERIODIC = "periodic"


class Side(Enum):
    """Which side of the domain."""

    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    TOP = "top"


@dataclass(frozen=True)
class SideBC:
    """Boundary condition on one side.

    ``velocity`` is the velocity *of the boundary itself* as ``(ux, uy)`` -- the
    lid-driven cavity's top side uses ``velocity=(1.0, 0.0)``.  ``scalar`` is the value
    imposed on scalars for ``INFLOW`` (ignored by other kinds).

    For ``INFLOW`` the *magnitude* of the normal component is used, signed so that the
    flow always enters the domain; the tangential part is taken from ``velocity``.
    """

    kind: Kind
    velocity: tuple[float, float] = (0.0, 0.0)
    scalar: float = 0.0

    def normal_into_domain(self, side: Side) -> float:
        """Normal (into-the-domain) speed and tangential speed for ``side``."""
        ux, uy = self.velocity
        if side in (Side.LEFT, Side.BOTTOM):
            return abs(ux if side is Side.LEFT else uy)
        return -abs(ux if side is Side.RIGHT else uy)

    def tangential(self, side: Side) -> float:
        """Tangential boundary speed for ``side``."""
        ux, uy = self.velocity
        return uy if side in (Side.LEFT, Side.RIGHT) else ux


@dataclass(frozen=True)
class BoundaryConditions:
    """Immutable set of boundary conditions for the four sides."""

    left: SideBC
    right: SideBC
    bottom: SideBC
    top: SideBC

    def side(self, side: Side) -> SideBC:
        match side:
            case Side.LEFT:
                return self.left
            case Side.RIGHT:
                return self.right
            case Side.BOTTOM:
                return self.bottom
            case Side.TOP:
                return self.top

    @property
    def is_x_periodic(self) -> bool:
        return self.left.kind is Kind.PERIODIC and self.right.kind is Kind.PERIODIC

    @property
    def is_y_periodic(self) -> bool:
        return self.bottom.kind is Kind.PERIODIC and self.top.kind is Kind.PERIODIC

    @property
    def has_open_boundary(self) -> bool:
        """True when at least one side is an outflow (the pressure Poisson problem then
        has a Dirichlet condition and is non-singular)."""
        return any(
            s.kind is Kind.OUTFLOW for s in (self.left, self.right, self.bottom, self.top)
        )

    def validate(self) -> BoundaryConditions:
        """Periodicity must come in pairs; raises ``ValueError`` otherwise."""
        if (self.left.kind is Kind.PERIODIC) != (self.right.kind is Kind.PERIODIC):
            raise ValueError("left/right must both be PERIODIC or neither")
        if (self.bottom.kind is Kind.PERIODIC) != (self.top.kind is Kind.PERIODIC):
            raise ValueError("bottom/top must both be PERIODIC or neither")
        return self


def periodic_bc() -> BoundaryConditions:
    """Periodic on all four sides."""
    p = SideBC(Kind.PERIODIC)
    return BoundaryConditions(p, p, p, p)


def walls_bc(velocity: tuple[float, float] = (0.0, 0.0)) -> BoundaryConditions:
    """No-slip on all four walls, with the top wall moving at ``velocity`` (the lid)."""
    fixed = SideBC(Kind.NO_SLIP)
    return BoundaryConditions(fixed, fixed, fixed, SideBC(Kind.NO_SLIP, velocity=velocity))


# --------------------------------------------------------------------------------------
# applying boundary conditions to the fields (in place)
# --------------------------------------------------------------------------------------
_SIDES = (Side.LEFT, Side.RIGHT, Side.BOTTOM, Side.TOP)


def apply_velocity_bc(
    u: Array, v: Array, bc: BoundaryConditions, *, inplace: bool = True
) -> tuple[Array, Array]:
    """Set the *normal* velocity on the domain faces.

    Wall (``NO_SLIP``/``FREE_SLIP``) faces get the wall's normal velocity -- zero unless
    the wall itself moves normally.  ``INFLOW`` faces get the prescribed normal velocity,
    ``OUTFLOW`` faces are untouched (zero-gradient outflow), ``PERIODIC`` faces are made
    equal (last face copied from the first).  Tangential components are *not* touched here:
    a staggered normal face carries no information about the tangential velocity, which is
    instead imposed through ghost values in :func:`pad_velocity`.
    """
    if not inplace:
        u = u.copy()
        v = v.copy()
    bc.validate()

    u[:, 0] = _normal_face_value(bc.left, Side.LEFT, u[:, 0])
    u[:, -1] = _normal_face_value(bc.right, Side.RIGHT, u[:, -1])
    v[0, :] = _normal_face_value(bc.bottom, Side.BOTTOM, v[0, :])
    v[-1, :] = _normal_face_value(bc.top, Side.TOP, v[-1, :])

    if bc.is_x_periodic:
        u[:, -1] = u[:, 0]
    if bc.is_y_periodic:
        v[-1, :] = v[0, :]
    return u, v


def _normal_face_value(sbc: SideBC, side: Side, current: Array) -> Array:
    """Value to write on a boundary face (the normal component)."""
    match sbc.kind:
        case Kind.NO_SLIP | Kind.FREE_SLIP:
            if side in (Side.LEFT, Side.RIGHT):
                return np.full_like(current, sbc.velocity[0])
            return np.full_like(current, sbc.velocity[1])
        case Kind.INFLOW:
            return np.full_like(current, sbc.normal_into_domain(side))
        case Kind.OUTFLOW | Kind.PERIODIC:
            return current


# --------------------------------------------------------------------------------------
# padding: ghost cells
# --------------------------------------------------------------------------------------
def pad_scalar(field: Array, bc: BoundaryConditions) -> Array:
    """Return a ``(ny+2, nx+2)`` copy of a cell-centred field with one ghost layer.

    ``NO_SLIP``/``FREE_SLIP``/``OUTFLOW`` use a zero-gradient ghost, ``INFLOW`` imposes
    ``SideBC.scalar`` and ``PERIODIC`` wraps.  Corner ghosts are the bilinear extrapolation
    ``out[1,0] + out[0,1] - out[1,1]`` so diagonal sampling stays sane.

    With shape ``(ny+2, nx+2)``, array index ``(j+1, i+1)`` is physical cell ``(j, i)``, so
    the correct sampling offsets are ``x0 = -h/2``, ``y0 = -h/2``.
    """
    ny, nx = field.shape
    out = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    out[1:-1, 1:-1] = field

    edges: dict[Side, tuple[Array, Array]] = {
        Side.LEFT: (field[:, 0], field[:, -1]),
        Side.RIGHT: (field[:, -1], field[:, 0]),
        Side.BOTTOM: (field[0, :], field[-1, :]),
        Side.TOP: (field[-1, :], field[0, :]),
    }
    for side, (interior, wrapped) in edges.items():
        sbc = bc.side(side)
        if sbc.kind is Kind.PERIODIC:
            g = wrapped
        elif sbc.kind is Kind.INFLOW:
            g = np.full_like(interior, sbc.scalar)
        else:
            g = interior
        if side is Side.LEFT:
            out[1:-1, 0] = g
        elif side is Side.RIGHT:
            out[1:-1, -1] = g
        elif side is Side.BOTTOM:
            out[0, 1:-1] = g
        else:
            out[-1, 1:-1] = g

    out[0, 0] = out[1, 0] + out[0, 1] - out[1, 1]
    out[0, -1] = out[1, -1] + out[0, -2] - out[1, -2]
    out[-1, 0] = out[-2, 0] + out[-1, 1] - out[-2, 1]
    out[-1, -1] = out[-2, -1] + out[-1, -2] - out[-2, -2]
    return out


def pad_velocity(u: Array, v: Array, bc: BoundaryConditions) -> tuple[Array, Array]:
    """Ghost-cell padded copies of ``u`` and ``v`` used by the advection backtrace.

    Index convention (``ny, nx = u.shape[0], v.shape[1]``)::

        up[j + 1, i + 1] = u[j, i]        up has shape (ny+2, nx+3)
        vp[j + 1, i + 1] = v[j, i]        vp has shape (ny+3, nx+2)

    so ``up`` index ``(j, i)`` sits at physical ``x = (i-1) h``, ``y = (j - 1/2) h``
    (sample it with ``x0=-h``, ``y0=-h/2``) and ``vp`` index ``(j, i)`` sits at
    ``x = (i - 1/2) h``, ``y = (j - 1) h`` (sample with ``x0=-h/2``, ``y0=-h``).

    * ``NO_SLIP`` / ``FREE_SLIP``: ghosts are reflected about the wall value,
      ``ghost = 2*wall_value - interior``, for **both** components.  For a tangential
      component this places the interpolated value *at* the wall exactly on the wall speed
      (zero for no-slip, the lid speed for the lid).  For a normal component it gives the
      odd mirror ``-interior`` at a wall at rest, so the interpolated normal velocity on
      the wall is exactly zero -- a backtrace that reaches the wall does not keep the
      overshooting velocity it had one cell inside.  The mirror partner of each ghost is
      the sample one cell *inside* the wall, not the wall sample itself.
    * ``INFLOW``: same reflection, about the prescribed inflow velocity.
    * ``OUTFLOW``: zero-gradient, the ghost copies the adjacent interior sample.
    * ``PERIODIC``: wrap by ``nx`` / ``ny`` samples, which is the true period of the padded
      index space (the periodic faces are stored twice, so the padded array repeats).
    """
    ny, nx = u.shape[0], v.shape[1]
    up = np.zeros((ny + 2, nx + 3), dtype=np.float64)
    vp = np.zeros((ny + 3, nx + 2), dtype=np.float64)
    up[1 : ny + 1, 1 : nx + 2] = u
    vp[1 : ny + 2, 1 : nx + 1] = v
    bc.validate()

    # ---- left / right: up ghost *columns* hold the normal (u) component,
    #      vp ghost *columns* hold the tangential (v) component
    for side, gcol_up, pcol_up, gcol_vp, pcol_vp in (
        (Side.LEFT, 0, 2, 0, 1),
        (Side.RIGHT, nx + 2, nx, nx + 1, nx),
    ):
        sbc = bc.side(side)
        if sbc.kind is Kind.PERIODIC:
            # period nx: index x = (i-1)h, so add or subtract nx columns
            up_col = nx if side is Side.LEFT else 2      # -h -> (nx-1)h ; (nx+1)h -> h
            vp_col = nx if side is Side.LEFT else 1
            up[:, gcol_up] = up[:, up_col]
            vp[:, gcol_vp] = vp[:, vp_col]
        elif sbc.kind is Kind.OUTFLOW:
            up[:, gcol_up] = up[:, 1 if side is Side.LEFT else nx + 1]
            vp[:, gcol_vp] = vp[:, 1 if side is Side.LEFT else nx]
        elif sbc.kind is Kind.FREE_SLIP:
            # Normal component vanishes at the wall; tangential component has zero
            # normal derivative (copy, rather than reflect, its interior partner).
            up[:, gcol_up] = -up[:, pcol_up]
            vp[:, gcol_vp] = vp[:, pcol_vp]
        else:
            up[:, gcol_up] = 2.0 * sbc.velocity[0] - up[:, pcol_up]
            vp[:, gcol_vp] = 2.0 * sbc.velocity[1] - vp[:, pcol_vp]

    # ---- bottom / top: vp ghost *rows* hold the normal (v) component,
    #      up ghost *rows* hold the tangential (u) component
    for side, grow_vp, prow_vp, grow_up, prow_up in (
        (Side.BOTTOM, 0, 2, 0, 1),
        (Side.TOP, ny + 2, ny, ny + 1, ny),
    ):
        sbc = bc.side(side)
        if sbc.kind is Kind.PERIODIC:
            # period ny rows: vp index y = (j-1)h ; up index y = (j-1/2)h
            vp_row = ny if side is Side.BOTTOM else 2
            up_row = ny if side is Side.BOTTOM else 1
            vp[grow_vp, :] = vp[vp_row, :]
            up[grow_up, :] = up[up_row, :]
        elif sbc.kind is Kind.OUTFLOW:
            vp[grow_vp, :] = vp[1 if side is Side.BOTTOM else ny + 1, :]
            up[grow_up, :] = up[1 if side is Side.BOTTOM else ny, :]
        elif sbc.kind is Kind.FREE_SLIP:
            vp[grow_vp, :] = -vp[prow_vp, :]
            up[grow_up, :] = up[prow_up, :]
        else:
            vp[grow_vp, :] = 2.0 * sbc.velocity[1] - vp[prow_vp, :]
            up[grow_up, :] = 2.0 * sbc.velocity[0] - up[prow_up, :]

    # ---- corners.  When both axes are periodic a corner is a pure periodic copy (shift by
    # the period in each direction); otherwise it is the average of the two adjacent ghost
    # strips.  Only diagonal sampling ever reaches these entries.
    xp, yp = bc.is_x_periodic, bc.is_y_periodic
    for arr in (up, vp):
        n_rows, n_cols = arr.shape
        for jc in (0, n_rows - 1):
            for ic in (0, n_cols - 1):
                jj = 1 if jc == 0 else n_rows - 2
                ii = 1 if ic == 0 else n_cols - 2
                if yp and xp:
                    jsrc = ny if jc == 0 else jc - ny
                    isrc = nx if ic == 0 else ic - nx
                    arr[jc, ic] = arr[jsrc, isrc]
                else:
                    arr[jc, ic] = 0.5 * (arr[jc, ii] + arr[jj, ic])
    return up, vp


# --------------------------------------------------------------------------------------
# discrete operators (defined here so every module shares one implementation)
# --------------------------------------------------------------------------------------
def divergence(u: Array, v: Array, h: float) -> Array:
    """Discrete divergence at cell centres, shape ``(ny, nx)``."""
    return (u[:, 1:] - u[:, :-1]) / h + (v[1:, :] - v[:-1, :]) / h


def gradient(
    p: Array, h: float, x_periodic: bool = False, y_periodic: bool = False
) -> tuple[Array, Array]:
    """Discrete pressure gradient on faces: ``(dp/dx on u faces, dp/dy on v faces)``.

    ``gx`` has shape ``(ny, nx+1)``: columns ``1 .. nx-1`` are the interior faces, column
    ``nx`` would be the face at ``x = lx`` (never used, since the last cell's right neighbour
    is a ghost) and column ``0`` does double duty -- it is the face at ``x = 0``, which is
    where the *periodic seam* lives.  When the x direction is periodic it carries the wrap
    difference ``(p[:, 0] - p[:, -1])/h``; otherwise it is zero, which is the mirrored-ghost
    Neumann condition.  The same holds for ``gy`` and ``y``.  Getting this wrong leaves the
    projected velocity divergence-free everywhere except the boundary column -- a bug that
    was caught exactly this way, see ``docs/milestones/M3.md``.
    """
    gx = np.zeros((p.shape[0], p.shape[1] + 1), dtype=np.float64)
    gy = np.zeros((p.shape[0] + 1, p.shape[1]), dtype=np.float64)
    gx[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / h
    gy[1:-1, :] = (p[1:, :] - p[:-1, :]) / h
    # Boundary faces.  The pressure lives in cell centres, so face 0 of ``gx`` sits exactly
    # on the domain boundary and face nx does not exist -- what *is* needed is the gradient
    # on the *periodic* seam face, which is the same face as face 0.  A wall (Neumann) gives
    # zero gradient; a periodic seam gives the wrap difference, which is what makes the
    # boundary columns of gx non-zero.
    if x_periodic:
        # the faces at x = 0 and x = lx are the same face, and they are stored twice
        gx[:, 0] = (p[:, 0] - p[:, -1]) / h
        gx[:, -1] = gx[:, 0]
    else:
        gx[:, 0] = 0.0
        gx[:, -1] = 0.0
    if y_periodic:
        gy[0, :] = (p[0, :] - p[-1, :]) / h
        gy[-1, :] = gy[0, :]
    else:
        gy[0, :] = 0.0
        gy[-1, :] = 0.0
    return gx, gy


def vorticity(
    u: Array,
    v: Array,
    h: float,
    x_periodic: bool = False,
    y_periodic: bool = False,
) -> Array:
    """Vorticity ``dv/dx - du/dy`` at cell corners, shape ``(ny+1, nx+1)``.

    Positive vorticity is counter-clockwise.  Corner ``(j, i)`` sits at ``x = i h``,
    ``y = j h``; the face values bracketing it horizontally are ``v[j, i-1]`` and
    ``v[j, i]`` (cell-centre columns), those bracketing it vertically are ``u[j-1, i]``
    and ``u[j, i]``.  Hence ``dv/dx`` is available at corners ``i = 1 .. nx-1`` and
    ``du/dy`` at corners ``j = 1 .. ny-1``; the missing edge corners use the periodic wrap
    when ``*_periodic`` is set and are left at the one-sided (zero) difference otherwise.
    """
    ny = u.shape[0]
    nx = v.shape[1]
    dvdx = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    dvdx[1:-1, 1:nx] = v[1:-1, 1:] - v[1:-1, :-1]
    dudy = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    dudy[1:ny, 1:-1] = u[1:, 1:-1] - u[:-1, 1:-1]
    if x_periodic:
        wrap = v[1:-1, 0] - v[1:-1, -1]
        dvdx[1:-1, 0] = wrap
        dvdx[1:-1, nx] = wrap
    if y_periodic:
        wrap_y = u[0, 1:-1] - u[-1, 1:-1]
        dudy[0, 1:-1] = wrap_y
        dudy[ny, 1:-1] = wrap_y
    return (dvdx - dudy) / h


def curl_streamfunction(psi_corners: Array, h: float) -> tuple[Array, Array]:
    """MAC velocity from a streamfunction sampled at cell *corners*.

    ``u = dpsi/dy`` on vertical faces and ``v = -dpsi/dx`` on horizontal faces, using the
    two corner values bracketing each face::

        u[j, i] = (psi[j+1, i] - psi[j, i]) / h
        v[j, i] = -(psi[j, i+1] - psi[j, i]) / h

    With ``psi_corners`` of shape ``(ny+1, nx+1)`` this produces ``u`` of shape
    ``(ny, nx+1)`` and ``v`` of shape ``(ny+1, nx)``, and the resulting field is *exactly*
    divergence free in the discrete MAC sense (the four terms of the discrete divergence
    of a cell cancel pairwise).  If ``psi`` vanishes on the boundary the normal velocity on
    that boundary is exactly zero too -- a convenient way to build wall-bounded test fields.
    """
    _ny1, _nx1 = psi_corners.shape
    u = (psi_corners[1:, :] - psi_corners[:-1, :]) / h
    v = -(psi_corners[:, 1:] - psi_corners[:, :-1]) / h
    return np.ascontiguousarray(u), np.ascontiguousarray(v)
