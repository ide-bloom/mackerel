"""Exact reference fields sampled on the MAC grid.

Every field here is a closed-form solution of the incompressible Euler/Navier-Stokes
equations on a periodic or bounded domain.  They are shared by the test suite, the
validation scripts and the browser comparison, so there is exactly one definition of each.

Notation: ``psi`` is the streamfunction, ``u = dpsi/dy``, ``v = -dpsi/dx``.
"""

from __future__ import annotations

import numpy as np

from mackerel.grid import Array, Grid, curl_streamfunction

__all__ = [
    "checkerboard_pressure_field",
    "double_shear_layer",
    "solid_body_rotation",
    "taylor_green",
    "taylor_green_kinetic_energy",
    "taylor_green_streamfunction",
    "taylor_green_velocity",
    "vortex_pair_streamfunction",
]


# --------------------------------------------------------------------------------------
# Taylor-Green vortex
# --------------------------------------------------------------------------------------
def taylor_green_streamfunction(grid: Grid, k: float = 2.0 * np.pi, amplitude: float = 1.0) -> Array:
    """Streamfunction at cell corners: ``psi = (A/k) sin(kx) sin(ky)``.

    Sampled at corners so that :func:`mackerel.grid.curl_streamfunction` yields a MAC
    velocity that is *exactly* divergence free and periodic.
    """
    X, Y = grid.corner_coords()
    return (amplitude / k) * np.sin(k * X) * np.sin(k * Y)


def taylor_green(grid: Grid, k: float = 2.0 * np.pi, amplitude: float = 1.0) -> tuple[Array, Array]:
    """Exact Taylor-Green velocity on the MAC grid.

    ``u = A sin(kx) cos(ky)``, ``v = -A cos(kx) sin(ky)`` on ``[0, 2pi/k]^2`` with a
    periodic boundary condition.  Obtained as the discrete curl of the corner
    streamfunction, so the sampled field is exactly (discretely) divergence free while
    matching the analytic field to second order in ``h``.
    """
    psi = taylor_green_streamfunction(grid, k=k, amplitude=amplitude)
    return curl_streamfunction(psi, grid.h)


def taylor_green_velocity(grid: Grid, x: Array, y: Array, k: float = 2.0 * np.pi,
                          amplitude: float = 1.0) -> tuple[Array, Array]:
    """Analytic Taylor-Green velocity at arbitrary points (for error norms)."""
    return amplitude * np.sin(k * x) * np.cos(k * y), -amplitude * np.cos(k * x) * np.sin(k * y)


def taylor_green_kinetic_energy(t: float, nu: float, amplitude: float = 1.0) -> float:
    """Exact kinetic energy of the Taylor-Green vortex: ``E(t) = E0 exp(-4 k^2 nu t)``.

    With ``k = 2 pi`` and ``A = 1`` the mean energy is ``A^2 / 4 = 0.25``, and the exact
    enstrophy is ``2 k^2 E(t)``.
    """
    k = 2.0 * np.pi
    return float(0.25 * amplitude**2 * np.exp(-4.0 * k**2 * nu * t))


def taylor_green_enstrophy(t: float, nu: float, amplitude: float = 1.0) -> float:
    """Exact enstrophy of the Taylor-Green vortex: ``Z(t) = 2 k^2 E(t)``."""
    return float(2.0 * (2.0 * np.pi) ** 2 * taylor_green_kinetic_energy(t, nu, amplitude))


# --------------------------------------------------------------------------------------
# solid-body rotation
# --------------------------------------------------------------------------------------
def solid_body_rotation(grid: Grid, omega: float = 1.0,
                        center: tuple[float, float] | None = None) -> tuple[Array, Array]:
    """Rigid rotation ``u = -omega (y - yc)``, ``v = omega (x - xc)`` on the MAC faces.

    Exactly divergence free, with uniform vorticity ``2 omega``.  Used to test that
    advection transports a profile around a closed loop without distortion.
    """
    cx, cy = center if center is not None else (0.5 * grid.lx, 0.5 * grid.ly)
    _Xu, Yu = grid.u_coords()
    u = -omega * (Yu - cy)
    Xv, _Yv = grid.v_coords()
    v = omega * (Xv - cx)
    return u, v


# --------------------------------------------------------------------------------------
# double shear layer (flow feature reference)
# --------------------------------------------------------------------------------------
def double_shear_layer(grid: Grid, rho: float = 30.0, delta: float = 0.05,
                      amplitude: float = 1.0) -> tuple[Array, Array]:
    """Brown & Minion (1995) double shear layer initial condition.

    ``u = tanh(rho (y - 1/4))`` for ``y <= 1/2``, ``tanh(rho (3/4 - y))`` above, plus the
    small perturbation ``v = delta sin(2 pi x)``.  Periodic in both directions.
    """
    _, Yu = grid.u_coords()
    u = np.where(Yu <= 0.5, np.tanh(rho * (Yu - 0.25)), np.tanh(rho * (0.75 - Yu)))
    u *= amplitude
    Xv, _ = grid.v_coords()
    v = delta * np.sin(2.0 * np.pi * Xv)
    return np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)


# --------------------------------------------------------------------------------------
# isolated vortex pair
# --------------------------------------------------------------------------------------
def vortex_pair_streamfunction(grid: Grid, strength: float = 0.05,
                               separation: float = 0.25) -> Array:
    """Streamfunction of two counter-rotating Gaussian vortices at cell corners."""
    X, Y = grid.corner_coords()
    cx, cy = 0.5 * grid.lx, 0.5 * grid.ly
    s = separation
    return strength * (
        np.exp(-((X - (cx - s)) ** 2 + (Y - cy) ** 2) / (2 * 0.05**2))
        - np.exp(-((X - (cx + s)) ** 2 + (Y - cy) ** 2) / (2 * 0.05**2))
    )


# --------------------------------------------------------------------------------------
# checkerboard pressure field (the staggered-grid demonstration)
# --------------------------------------------------------------------------------------
def checkerboard_pressure_field(grid: Grid, amplitude: float = 1.0) -> Array:
    """The worst-case pressure mode for a *collocated* grid: ``p = (-1)^(i+j)``.

    On a collocated grid the centred difference of this mode is identically zero, so it is
    invisible to the pressure gradient and can grow without bound (checkerboarding).  On a
    staggered grid the same mode produces the *largest* possible face gradient -- which is
    why staggering kills the null space.  This function returns the cell-centred field; the
    gradients on either layout are computed in the tests and in ``docs/MATH.md``.
    """
    j, i = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
    return np.asarray(amplitude * (-1.0) ** (i + j), dtype=np.float64)
