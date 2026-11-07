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
from typing import Any

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
    # Optional conservative paired reseeding.  Opt-in so documented legacy validation stays
    # bit-for-bit reproducible.  ``reseed_target`` is the density a donor cell is drained back
    # to; ``None`` means the nominal ``particles_per_cell`` of the initial lattice.
    reseed: bool = False
    reseed_interval: int = 1
    min_particles_per_cell: int = 2
    max_particles_per_cell: int = 16
    reseed_target: int | None = None
    reseed_interior_only: bool = True
    reseed_radius: float = 4.0
    reseed_cooldown: int = 10
    reseed_seed: int = 0


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
        self.reseed_history: list[dict[str, Any]] = []
        self._reseed_rng = np.random.default_rng(self.config.reseed_seed)
        # Steps remaining before a relocated particle may be relocated again.  Without this a
        # particle can be shuttled between starved cells every step, destroying the Lagrangian
        # history that makes FLIP low-diffusion in the first place.
        self._reseed_cooldown = np.zeros(len(self.particles), dtype=np.intp)
        # These bounds only matter when reseeding is actually enabled.  Validating them
        # unconditionally rejected legitimate configurations such as a 9-particle-per-cell
        # seeding with the default 8-particle donor cap, even though nothing was reseeded.
        if self.config.reseed:
            if self.config.reseed_interval < 1:
                raise ValueError("reseed_interval must be positive")
            if self.config.reseed_cooldown < 0:
                raise ValueError("reseed_cooldown must be non-negative")
            if self.config.reseed_radius <= 0.0:
                raise ValueError("reseed_radius must be positive")
            if not 1 <= self.config.min_particles_per_cell <= self.config.max_particles_per_cell:
                raise ValueError("particle reseeding bounds must satisfy 1 <= min <= max")
            if self.reseed_target() > self.config.max_particles_per_cell:
                raise ValueError("reseed_target must not exceed max_particles_per_cell")

    def reseed_target(self) -> int:
        """Density a donor cell is drained back to (``particles_per_cell`` by default)."""
        target = self.config.reseed_target
        return self.particles_per_cell if target is None else target

    def _particle_counts(self) -> npt.NDArray[np.intp]:
        i = np.clip((self.particles.x / self.grid.h).astype(np.intp), 0, self.grid.nx - 1)
        j = np.clip((self.particles.y / self.grid.h).astype(np.intp), 0, self.grid.ny - 1)
        counts = np.zeros(self.grid.shape_cell, dtype=np.intp)
        np.add.at(counts, (j, i), 1)
        return counts

    def _particle_cells(self) -> BoolArray:
        return self._particle_counts() > 0

    def density_metrics(self, mask: BoolArray | None = None) -> dict[str, float]:
        """Particle sampling density on ``mask`` (default: cells containing liquid).

        The error target is the configured particles per cell, not a fitted density.  Passing
        an explicit mask lets a before/after comparison cover exactly the same cells.  A
        free-surface cell that the interface only partly crosses legitimately holds fewer than
        ``particles_per_cell`` particles, so ``rms_target_error`` mixes a real interior defect
        with that geometric fringe effect; :meth:`interior_deficits` separates the two.
        """
        counts = self._particle_counts()
        selected = counts > 0 if mask is None else mask
        values = counts[selected].astype(np.float64)
        target = float(self.particles_per_cell)
        stats = {
            "cells": float(values.size), "particles": float(np.sum(values)),
            "mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0,
            "rms_target_error": 0.0,
            "underfilled": float(np.count_nonzero(values < self.config.min_particles_per_cell)),
            "over_target": float(np.count_nonzero(values > self.reseed_target())),
            "over_max": float(np.count_nonzero(values > self.config.max_particles_per_cell)),
            "empty": float(np.count_nonzero(values == 0.0)),
        }
        if values.size == 0:
            return stats
        mean = float(np.mean(values))
        stats.update(
            mean=mean, std=float(np.std(values)),
            cv=float(np.std(values) / max(mean, 1e-30)),
            min=float(np.min(values)), max=float(np.max(values)),
            rms_target_error=float(np.sqrt(np.mean((values - target) ** 2))),
        )
        return stats

    def particle_cell_ids(self) -> npt.NDArray[np.intp]:
        """Flat ``j*nx + i`` index of the cell containing each particle."""
        g = self.grid
        i = np.clip((self.particles.x / g.h).astype(np.intp), 0, g.nx - 1)
        j = np.clip((self.particles.y / g.h).astype(np.intp), 0, g.ny - 1)
        return (j * g.nx + i).astype(np.intp)

    def _neighbour_count(self, mask: BoolArray) -> npt.NDArray[np.intp]:
        """Number of in-domain four-neighbours of each cell where ``mask`` is true."""
        count = np.zeros(mask.shape, dtype=np.intp)
        count[1:, :] += mask[:-1, :]
        count[:-1, :] += mask[1:, :]
        count[:, 1:] += mask[:, :-1]
        count[:, :-1] += mask[:, 1:]
        return count

    def _neighbour_total(self) -> npt.NDArray[np.intp]:
        """Number of in-domain four-neighbours of each cell (4, 3 at an edge, 2 at a corner)."""
        total = np.full(self.grid.shape_cell, 4, dtype=np.intp)
        total[0, :] -= 1
        total[-1, :] -= 1
        total[:, 0] -= 1
        total[:, -1] -= 1
        return total

    def _fringe_cells(self, occupied: BoolArray) -> BoolArray:
        """Liquid cells that touch the exterior air region (or the boundary)."""
        return occupied & (self._neighbour_count(occupied) < self._neighbour_total())

    def interior_deficits(self) -> dict[str, float]:
        """Count sampling defects that are *not* explained by the free surface.

        A hole is an empty cell enclosed by liquid; it is treated as air by the free-surface
        projection even though liquid surrounds it, which is the defect reseeding exists to
        remove.  A starved interior cell is an occupied cell below ``min_particles_per_cell``
        that has no exterior-air neighbour.  Starved *fringe* cells are reported separately
        because the interface legitimately crosses only part of such a cell.
        """
        counts = self._particle_counts()
        occupied = counts > 0
        inner = self._neighbour_count(occupied) == self._neighbour_total()
        holes = (~occupied) & inner
        starved = occupied & (counts < self.config.min_particles_per_cell)
        return {
            "interior_holes": float(np.count_nonzero(holes)),
            "interior_starved": float(np.count_nonzero(starved & inner)),
            "fringe_starved": float(np.count_nonzero(starved & ~inner)),
        }

    def _reseed_recipients(self, counts: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Cells eligible to receive a particle.

        By default these are cells whose sparsity is unambiguous: an empty cell that liquid
        fully surrounds, and a cell below ``min_particles_per_cell`` with no exterior-air
        neighbour.  Both rules require *all* in-domain four-neighbours to be liquid, so a
        refill can never add water beyond the free surface.
        ``reseed_interior_only=False`` also treats fringe cells as defects, which is measured by
        ``validation/flip_reseed.py`` because it mostly moves particles across the interface.
        """
        occupied = counts > 0
        inner = self._neighbour_count(occupied) == self._neighbour_total()
        holes = (~occupied) & inner
        starved = occupied & (counts < self.config.min_particles_per_cell)
        if self.config.reseed_interior_only:
            starved &= inner
        return np.flatnonzero((starved | holes).ravel())

    def _morton_codes(self, cells: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Locality-preserving Z-order key for flat cell indices.

        Donors and recipients are both sorted by this key before pairing, so an exchange
        usually happens between nearby cells instead of teleporting particles across the box.
        Ties break on the flat index, which keeps the whole scheme deterministic.
        """
        g = self.grid
        i = cells % g.nx
        j = cells // g.nx
        bits = max(1, int(np.ceil(np.log2(max(g.nx, g.ny)))))
        code = np.zeros(cells.size, dtype=np.intp)
        for b in range(bits):
            code |= ((i >> b) & 1) << np.intp(2 * b)
            code |= ((j >> b) & 1) << np.intp(2 * b + 1)
        return code

    def _donor_ranking(self) -> npt.NDArray[np.intp]:
        """Particle order that decides which members leave an overfull cell.

        Particles still under a post-move cooldown come last, then slowest first, so a donor
        keeps its fastest and least recently relocated members.  These are deterministic
        criteria, not random picks; ties fall back to the particle index.
        """
        p = self.particles
        speed = p.u * p.u + p.v * p.v
        return np.lexsort(
            (np.arange(len(p)), speed, self.particle_cell_ids(), self._reseed_cooldown > 0)
        )

    def reseed(self) -> dict[str, Any]:
        """Redistribute particles from overfull cells to starved interior cells and holes.

        The operation is *paired*: particles are moved, never created or destroyed.  Particle
        count, nominal volume and every particle's velocity are conserved exactly, so only
        positions change.  Cells holding more than :meth:`reseed_target` particles donate their
        surplus; see :meth:`interior_deficits` for the recipient rule.  Matched before/after
        density metrics are recorded so the improvement is measured rather than assumed.

        When demand exceeds the available surplus, one seeded binomial keep probability scales
        every recipient, so no donor is over-drained and the outcome stays reproducible.  Each
        relocated particle then enters a cooldown, which stops a particle from being shuttled
        between starved cells every step and destroying its Lagrangian history.
        """
        counts = self._particle_counts()
        before_deficits = self.interior_deficits()
        mask = counts > 0
        before = self.density_metrics(mask)
        record: dict[str, Any] = {
            "time": float(self.time), "moved": 0.0, "keep_probability": 1.0,
            "donor_cells": 0.0, "recipient_cells": 0.0,
            "particle_count": float(len(self.particles)),
            "mean_move_distance": 0.0, "max_move_distance": 0.0, "mean_moved_speed": 0.0,
            "before": before, "after": before, "deficits_before": before_deficits,
        }
        flat = counts.ravel()
        target = self.reseed_target()
        overfull = np.flatnonzero(flat > target)
        recipients = self._reseed_recipients(counts)
        if overfull.size and recipients.size and len(self.particles):
            record.update(self._move_particles(flat, overfull, recipients, target, record))
        self.fluid = self._particle_cells()
        record["after"] = self.density_metrics(mask)
        record["deficits_after"] = self.interior_deficits()
        self.reseed_history.append(record)
        return record

    def _pair_moves(
        self,
        donor_order: npt.NDArray[np.intp],
        slots: npt.NDArray[np.intp],
        radius: float,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
        """Greedily pair each receiving slot with the nearest still-available donor particle.

        Donors are ranked by distance and then by particle index, so the pairing is
        deterministic.  A donor outside ``radius`` is not used at all: a local refill is the
        point of reseeding, and a long-range jump would fabricate a particle at a location whose
        velocity history it never had.  Returns the chosen particles, their target cells and the
        number of slots left unfilled because no donor was close enough.
        """
        h = self.grid.h
        target_i = (slots % self.grid.nx).astype(np.float64) + 0.5
        target_j = (slots // self.grid.nx).astype(np.float64) + 0.5
        px, py = self.particles.x[donor_order], self.particles.y[donor_order]
        available = np.ones(donor_order.size, dtype=bool)
        chosen: list[int] = []
        filled: list[int] = []
        for slot, (ci, cj) in enumerate(zip(target_i, target_j, strict=True)):
            if not available.any():
                break
            distance = np.hypot(px - ci * h, py - cj * h)
            distance[~available] = np.inf
            nearest = int(np.argmin(distance))
            if distance[nearest] > radius * h:
                continue
            available[nearest] = False
            chosen.append(nearest)
            filled.append(slot)
        return (
            donor_order[np.asarray(chosen, dtype=np.intp)],
            slots[np.asarray(filled, dtype=np.intp)],
            int(slots.size - len(chosen)),
        )

    def _move_particles(
        self,
        flat: npt.NDArray[np.intp],
        overfull: npt.NDArray[np.intp],
        recipients: npt.NDArray[np.intp],
        target: int,
        record: dict[str, Any],
    ) -> dict[str, float]:
        """Perform the actual exchange for :meth:`reseed` and report what changed."""
        surplus = int(np.sum(flat[overfull] - target))
        need = np.maximum(self.config.min_particles_per_cell - flat[recipients], 0)
        need = np.minimum(need, target)  # never fill a recipient above the nominal density
        keep = 1.0 if int(need.sum()) <= surplus else surplus / int(need.sum())
        if keep < 1.0:
            need = self._reseed_rng.binomial(need, keep)
        n_slots = int(min(need.sum(), surplus))
        updates: dict[str, float] = {
            "keep_probability": float(keep), "requested_slots": float(need.sum()),
        }
        if n_slots == 0:
            return updates

        cell_of_particle = self.particle_cell_ids()
        order = self._donor_ranking()
        ranked_cells = cell_of_particle[order]
        starts = np.searchsorted(ranked_cells, np.arange(self.grid.nx * self.grid.ny + 1))
        rank = np.empty(len(self.particles), dtype=np.intp)
        rank[order] = np.arange(len(self.particles)) - starts[ranked_cells]
        in_rank = rank < np.maximum(flat - target, 0)[cell_of_particle]
        donor_order = order[in_rank[order]]
        slots = np.repeat(recipients, need)[:n_slots]
        moved, targets, unfilled = self._pair_moves(
            donor_order, slots, float(self.config.reseed_radius)
        )
        updates.update(donor_cells=float(overfull.size),
                       recipient_cells=float(recipients.size),
                       unfilled_slots=float(unfilled))
        if moved.size == 0:
            return updates

        h = self.grid.h
        ti = (targets % self.grid.nx).astype(np.float64)
        tj = (targets // self.grid.nx).astype(np.float64)
        new_x = (ti + 0.15 + 0.7 * self._reseed_rng.random(moved.size)) * h
        new_y = (tj + 0.15 + 0.7 * self._reseed_rng.random(moved.size)) * h
        p = self.particles
        distance = np.hypot(new_x - p.x[moved], new_y - p.y[moved])
        updates.update(
            moved=float(moved.size),
            mean_move_distance=float(np.mean(distance)),
            max_move_distance=float(np.max(distance)),
            mean_moved_speed=float(np.mean(np.hypot(p.u[moved], p.v[moved]))),
        )
        p.x[moved] = new_x
        p.y[moved] = new_y
        self._reseed_cooldown[moved] = self.config.reseed_cooldown + 1
        return updates

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
        self._reseed_cooldown -= 1
        np.clip(self._reseed_cooldown, 0, None, out=self._reseed_cooldown)
        self.fluid = self._particle_cells()
        if self.config.reseed and (len(self.history) + 1) % self.config.reseed_interval == 0:
            self.reseed()
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
