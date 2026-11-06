"""Diagnostics: divergence, kinetic energy, vorticity, enstrophy, CFL, total scalar.

All integrals use the midpoint rule on cells of area ``h^2`` and density 1, so
"energy" means ``sum(u^2 + v^2)/2 * h^2`` with ``u``/``v`` evaluated at cell centres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mackerel.grid import Array, BoundaryConditions, Grid, divergence, vorticity

__all__ = [
    "Diagnostics",
    "diagnose",
    "divergence_field",
    "divergence_norms",
    "enstrophy",
    "kinetic_energy",
    "max_cfl",
    "max_speed",
    "scalar_integral",
    "total_scalar",
    "vorticity_field",
]


# --------------------------------------------------------------------------------------
# divergence
# --------------------------------------------------------------------------------------
def divergence_field(u: Array, v: Array, grid: Grid) -> Array:
    """Discrete divergence at cell centres, shape ``(ny, nx)``."""
    return divergence(u, v, grid.h)


def divergence_norms(div: Array) -> tuple[float, float]:
    """``(max |div|, L2 norm of div)`` where the L2 norm is ``sqrt(sum(div^2 h^2))``.

    The caller passes the grid-free array; use :func:`divergence_norms_on_grid` when the
    physical scaling matters.  This version returns the plain 2-norm of the array.
    """
    return float(np.max(np.abs(div))), float(np.sqrt(np.sum(div * div)))


def divergence_norms_on_grid(div: Array, grid: Grid) -> tuple[float, float]:
    """``(max |div|, L2 norm)`` with ``L2 = sqrt(sum(div^2) h^2)``."""
    return float(np.max(np.abs(div))), float(np.sqrt(np.sum(div * div) * grid.cell_area))


# --------------------------------------------------------------------------------------
# energy / vorticity / enstrophy
# --------------------------------------------------------------------------------------
def kinetic_energy(u: Array, v: Array, grid: Grid) -> float:
    """Total kinetic energy ``sum_cell 0.5 (u_c^2 + v_c^2) h^2`` (density 1)."""
    uc = grid.u_to_centers(u)
    vc = grid.v_to_centers(v)
    return float(0.5 * np.sum(uc * uc + vc * vc) * grid.cell_area)


def vorticity_field(u: Array, v: Array, grid: Grid, bc: BoundaryConditions | None = None) -> Array:
    """Vorticity ``dv/dx - du/dy`` at cell corners, shape ``(ny+1, nx+1)``.

    Boundary corners honour the periodic ghost values when ``bc`` is supplied and periodic.
    """
    xp = bool(bc is not None and bc.is_x_periodic)
    yp = bool(bc is not None and bc.is_y_periodic)
    return vorticity(u, v, grid.h, x_periodic=xp, y_periodic=yp)


def enstrophy(u: Array, v: Array, grid: Grid, bc: BoundaryConditions | None = None) -> float:
    """Total enstrophy ``sum 0.5 omega^2`` (cells of area ``h^2``).

    Vorticity lives on corners; each interior corner is taken to represent one cell, so the
    sum over the interior corner array times ``h^2`` is the midpoint approximation.
    """
    w = vorticity_field(u, v, grid, bc)[1:-1, 1:-1]
    return float(0.5 * np.sum(w * w) * grid.cell_area)


def scalar_integral(field: Array, grid: Grid) -> float:
    """Midpoint-rule integral of a cell-centred field."""
    return float(np.sum(field) * grid.cell_area)


def total_scalar(field: Array, grid: Grid) -> float:
    """Alias of :func:`scalar_integral` -- total amount of dye/temperature/... ."""
    return scalar_integral(field, grid)


# --------------------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------------------
def max_speed(u: Array, v: Array, grid: Grid) -> float:
    """Largest speed magnitude at cell centres."""
    uc = grid.u_to_centers(u)
    vc = grid.v_to_centers(v)
    return float(np.sqrt(np.max(uc * uc + vc * vc)))


def max_cfl(u: Array, v: Array, grid: Grid, dt: float) -> float:
    """``dt * max(|u| + |v|) / h`` -- the advective CFL number (should stay below ~1)."""
    uc = np.abs(grid.u_to_centers(u))
    vc = np.abs(grid.v_to_centers(v))
    return float(dt * np.max(uc + vc) / grid.h)


# --------------------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Diagnostics:
    """Snapshot of the scalar diagnostics of a velocity field."""

    div_max: float
    div_l2: float
    kinetic_energy: float
    enstrophy: float
    cfl: float
    max_speed: float


def diagnose(u: Array, v: Array, grid: Grid, dt: float = 0.0,
             bc: BoundaryConditions | None = None) -> Diagnostics:
    """Compute every scalar diagnostic in one call."""
    div = divergence_field(u, v, grid)
    dmax, dl2 = divergence_norms_on_grid(div, grid)
    return Diagnostics(
        div_max=dmax,
        div_l2=dl2,
        kinetic_energy=kinetic_energy(u, v, grid),
        enstrophy=enstrophy(u, v, grid, bc),
        cfl=max_cfl(u, v, grid, dt),
        max_speed=max_speed(u, v, grid),
    )


# --------------------------------------------------------------------------------------
# environment / dependency reporting
# --------------------------------------------------------------------------------------
def ffmpeg_report() -> str:
    """Human-readable dependency report for the ``mackerel info`` command."""
    import matplotlib
    import numpy
    import PIL
    import scipy

    from mackerel.render import ffmpeg_available

    lines = [
        f"numpy      {numpy.__version__}",
        f"scipy      {scipy.__version__}",
        f"matplotlib {matplotlib.__version__}",
        f"pillow     {PIL.__version__}",
    ]
    try:
        import numba

        lines.append(f"numba      {numba.__version__}")
    except ImportError:  # pragma: no cover - optional dependency
        lines.append("numba      not installed")
    lines.append(f"ffmpeg     {'found' if ffmpeg_available() else 'not found (GIF only)'}")
    return "\n".join(lines)
