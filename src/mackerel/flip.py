"""PIC/FLIP free-surface water on the MAC grid.

Particles carry mass and velocity.  Each step transfers particle velocity to the two MAC face
sets with bilinear weights, applies gravity, solves a pressure system in the liquid, and then
updates particle velocities either directly (PIC) or by the grid velocity increment (FLIP).
The ``pic_blend`` parameter is the PIC fraction: 1 is pure PIC, 0 is pure FLIP, and .05 is a
usual low-diffusion FLIP blend.

Air is not a solid obstacle.  Liquid-air faces see atmospheric pressure ``p=0``; the
``free_surface=True`` path in :mod:`mackerel.project` builds that Dirichlet contribution and
only corrects faces touching liquid.  No particle is deleted at a wall, so a closed dam-break
experiment retains its particle count and water volume up to the grid transfer error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mackerel.grid import Array, BoolArray, BoundaryConditions, Grid, apply_velocity_bc, walls_bc
from mackerel.project import ProjectionResult, project_velocity

__all__ = ["FreeSurfaceWater", "ParticleSet", "WaterConfig", "WaterStepInfo", "dam_break_particles"]

IntArray = npt.NDArray[np.intp]


@dataclass
class ParticleSet:
    """Structure-of-arrays particle storage, all float64."""

    x: Array
    y: Array
    u: Array
    v: Array

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.u = np.asarray(self.u, dtype=np.float64)
        self.v = np.asarray(self.v, dtype=np.float64)
        if not (self.x.shape == self.y.shape == self.u.shape == self.v.shape):
            raise ValueError("particle arrays must have identical shapes")

    def __len__(self) -> int:
        return int(self.x.size)

    def copy(self) -> ParticleSet:
        return ParticleSet(self.x.copy(), self.y.copy(), self.u.copy(), self.v.copy())

    @classmethod
    def from_region(
        cls,
        grid: Grid,
        region: Callable[[Array, Array], BoolArray],
        particles_per_cell: int = 4,
        seed: int = 0,
    ) -> ParticleSet:
        """Seed a jittered lattice in cells whose centre satisfies ``region(X, Y)``."""
        if particles_per_cell < 1:
            raise ValueError("particles_per_cell must be positive")
        side = int(np.ceil(np.sqrt(particles_per_cell)))
        X, Y = grid.cell_coords()
        jj, ii = np.nonzero(region(X, Y))
        rng = np.random.default_rng(seed)
        xs: list[Array] = []
        ys: list[Array] = []
        for k in range(particles_per_cell):
            ox = (k % side + 0.5) / side
            oy = (k // side + 0.5) / side
            jitter = 0.1 / side
            xs.append((ii + ox + rng.uniform(-jitter, jitter, len(ii))) * grid.h)
            ys.append((jj + oy + rng.uniform(-jitter, jitter, len(ii))) * grid.h)
        x = np.concatenate(xs) if xs else np.empty(0, dtype=np.float64)
        y = np.concatenate(ys) if ys else np.empty(0, dtype=np.float64)
        return cls(x, y, np.zeros_like(x), np.zeros_like(y))


@dataclass(frozen=True)
class WaterConfig:
    dt: float = 0.005
    cfl: float = 0.5
    gravity: float = 9.81
    pic_blend: float = 0.05
    solver: str = "conjugate_gradient"
    pressure_tol: float = 1e-8
    pressure_maxiter: int = 1000
    transfer_backend: str = "numpy"


@dataclass(frozen=True)
class WaterStepInfo:
    time: float
    dt: float
    liquid_cells: int
    pressure_iterations: int
    pressure_residual: float
    divergence: float
    max_speed: float


def _indices_weights(
    x: Array, y: Array, h: float, shape: tuple[int, int], component: str
) -> tuple[IntArray, IntArray, Array, Array, Array, Array]:
    """Four-corner indices/weights for a particle sampled on a MAC component."""
    ny, nx = shape
    if component == "u":
        fx, fy = x / h, y / h - 0.5
    elif component == "v":
        fx, fy = x / h - 0.5, y / h
    else:
        fx, fy = x / h - 0.5, y / h - 0.5
    i0 = np.clip(np.floor(fx).astype(np.intp), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(np.intp), 0, ny - 2)
    # Clamping an index without clamping its fraction creates negative/oversized weights
    # for particles within half a cell of a domain wall.  Both must be clamped together.
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)
    return i0, j0, tx, ty, 1.0 - tx, 1.0 - ty


def _scatter(
    values: Array, x: Array, y: Array, h: float, shape: tuple[int, int], component: str
) -> tuple[Array, Array]:
    """Bilinear particle-to-grid sum and weight sum using vectorised ``np.add.at``."""
    i0, j0, tx, ty, wx, wy = _indices_weights(x, y, h, shape, component)
    out = np.zeros(shape, dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)
    for dj, di, w in (
        (0, 0, wx * wy), (0, 1, tx * wy), (1, 0, wx * ty), (1, 1, tx * ty)
    ):
        np.add.at(out, (j0 + dj, i0 + di), w * values)
        np.add.at(weight, (j0 + dj, i0 + di), w)
    return out, weight


def _gather(values: Array, x: Array, y: Array, h: float, component: str) -> Array:
    """Bilinear grid-to-particle interpolation on one MAC component."""
    i0, j0, tx, ty, wx, wy = _indices_weights(x, y, h, values.shape, component)
    return (
        wx * wy * values[j0, i0]
        + tx * wy * values[j0, i0 + 1]
        + wx * ty * values[j0 + 1, i0]
        + tx * ty * values[j0 + 1, i0 + 1]
    )


def _normalise(sum_values: Array, weights: Array) -> Array:
    return np.divide(sum_values, weights, out=np.zeros_like(sum_values), where=weights > 1e-12)


@dataclass
class FreeSurfaceWater:
    """A PIC/FLIP liquid in a rectangular domain."""

    grid: Grid
    particles: ParticleSet
    bc: BoundaryConditions = field(default_factory=walls_bc)
    config: WaterConfig = field(default_factory=WaterConfig)
    time: float = 0.0
    particles_per_cell: int = 4

    def __post_init__(self) -> None:
        self.bc.validate()
        if self.particles_per_cell < 1:
            raise ValueError("particles_per_cell must be positive")
        if self.config.transfer_backend not in {"numpy", "numba"}:
            raise ValueError("transfer_backend must be 'numpy' or 'numba'")
        self.u = self.grid.zeros_u()
        self.v = self.grid.zeros_v()
        self.fluid = self._particle_cells()
        self._initial_particle_count = len(self.particles)
        self._initial_volume = self._initial_particle_count * self.grid.cell_area / self.particles_per_cell
        self.last_projection: ProjectionResult | None = None
        self.history: list[WaterStepInfo] = []

    def _particle_cells(self) -> BoolArray:
        i = np.clip((self.particles.x / self.grid.h).astype(np.intp), 0, self.grid.nx - 1)
        j = np.clip((self.particles.y / self.grid.h).astype(np.intp), 0, self.grid.ny - 1)
        cells = np.zeros(self.grid.shape_cell, dtype=bool)
        np.add.at(cells, (j, i), True)
        return cells

    def _transfer_to_grid(self) -> tuple[Array, Array]:
        p = self.particles
        scatter = _scatter
        if self.config.transfer_backend == "numba":
            from mackerel.flip_numba import scatter_numba

            scatter = scatter_numba
        su, wu = scatter(p.u, p.x, p.y, self.grid.h, self.grid.shape_u, "u")
        sv, wv = scatter(p.v, p.x, p.y, self.grid.h, self.grid.shape_v, "v")
        self.u = _normalise(su, wu)
        self.v = _normalise(sv, wv)
        apply_velocity_bc(self.u, self.v, self.bc)
        return wu, wv

    def dt_for_cfl(self) -> float:
        speed = float(max(np.max(np.abs(self.particles.u), initial=0.0),
                          np.max(np.abs(self.particles.v), initial=0.0)))
        if speed < 1e-12:
            return self.config.dt
        return min(self.config.dt, self.config.cfl * self.grid.h / speed)

    def step(self, dt: float | None = None) -> WaterStepInfo:
        """Advance particles and the free-surface velocity field by one step."""
        if len(self.particles) == 0:
            raise ValueError("cannot step an empty particle set")
        step_dt = self.dt_for_cfl() if dt is None else float(dt)
        if step_dt <= 0.0:
            raise ValueError("dt must be positive")
        self.fluid = self._particle_cells()
        _wu, wv = self._transfer_to_grid()
        old_u = self.u.copy()  # the old *grid* field for the FLIP increment
        old_v = self.v.copy()
        self.v[wv > 1e-12] -= self.config.gravity * step_dt
        apply_velocity_bc(self.u, self.v, self.bc)

        projection = project_velocity(
            self.u, self.v, self.grid, self.bc, step_dt,
            solver=self.config.solver, tol=self.config.pressure_tol,
            maxiter=self.config.pressure_maxiter, solid=~self.fluid,
            free_surface=True, keep_solid_velocity=True,
        )
        self.last_projection = projection
        self.u, self.v = projection.u, projection.v
        p = self.particles
        new_up = _gather(self.u, p.x, p.y, self.grid.h, "u")
        new_vp = _gather(self.v, p.x, p.y, self.grid.h, "v")
        old_up = _gather(old_u, p.x, p.y, self.grid.h, "u")
        old_vp = _gather(old_v, p.x, p.y, self.grid.h, "v")
        blend = self.config.pic_blend
        p.u = blend * new_up + (1.0 - blend) * (p.u + new_up - old_up)
        p.v = blend * new_vp + (1.0 - blend) * (p.v + new_vp - old_vp)
        p.x += step_dt * p.u
        p.y += step_dt * p.v
        # Closed-wall particle collisions: do not lose water at an artificial outflow.
        eps = 1e-8 * self.grid.h
        left = p.x < eps
        right = p.x > self.grid.lx - eps
        bottom = p.y < eps
        top = p.y > self.grid.ly - eps
        p.u[left | right] = 0.0
        p.v[bottom | top] = 0.0
        p.x = np.clip(p.x, eps, self.grid.lx - eps)
        p.y = np.clip(p.y, eps, self.grid.ly - eps)
        self.time += step_dt
        self.fluid = self._particle_cells()
        info = WaterStepInfo(
            time=self.time, dt=step_dt, liquid_cells=int(np.count_nonzero(self.fluid)),
            pressure_iterations=projection.iterations,
            pressure_residual=projection.residual, divergence=projection.div_max,
            max_speed=float(max(np.max(np.abs(p.u)), np.max(np.abs(p.v)))),
        )
        self.history.append(info)
        return info

    def run_steps(self, steps: int) -> list[WaterStepInfo]:
        return [self.step() for _ in range(steps)]

    def run_until(self, t_end: float) -> list[WaterStepInfo]:
        """Advance to ``t_end``, clamping the final adaptive step exactly."""
        records: list[WaterStepInfo] = []
        while self.time < t_end - 1e-14:
            records.append(self.step(min(self.dt_for_cfl(), t_end - self.time)))
        return records

    def front_position(self) -> float:
        """Rightmost particle x, a direct dam-break front measurement."""
        return float(np.max(self.particles.x))

    def water_volume(self) -> float:
        """Nominal liquid volume; closed-wall particle transport conserves it exactly."""
        if self._initial_particle_count == 0:
            return 0.0
        return self._initial_volume * len(self.particles) / self._initial_particle_count


def dam_break_particles(grid: Grid, width: float = 0.25, height: float = 0.5) -> ParticleSet:
    """Four particles per cell in a rectangular initially still water column."""
    return ParticleSet.from_region(
        grid, lambda X, Y: (width >= X) & (height >= Y), particles_per_cell=4
    )
