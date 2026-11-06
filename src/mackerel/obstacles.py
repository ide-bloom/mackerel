"""Immersed solid obstacles on a MAC grid.

A solid is represented by a boolean value at each cell centre.  Every MAC face touching a
solid cell is a solid face: its normal/tangential velocity is prescribed (zero for a fixed
obstacle, or the obstacle's translational velocity for a moving one).  The pressure system
uses the same mask, so there is no pressure gradient across those faces.

This is a staircase immersed-boundary method, not a body-fitted cylinder mesh.  It is
portable, works with the existing matrix-free solvers, and its geometric error is reduced by
refining the grid.  The validation report measures the resulting wake rather than claiming
smooth-boundary accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mackerel.grid import Array, BoolArray, Grid

__all__ = [
    "MovingObstacle",
    "apply_solid_velocity",
    "box_mask",
    "circle_mask",
    "solid_face_masks",
]


def circle_mask(grid: Grid, center: tuple[float, float], radius: float) -> BoolArray:
    """Cell-centred staircase mask for a circle."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    X, Y = grid.cell_coords()
    return np.asarray((X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2, dtype=bool)


def box_mask(
    grid: Grid, lower: tuple[float, float], upper: tuple[float, float]
) -> BoolArray:
    """Cell-centred mask for an axis-aligned rectangle, with inclusive lower/upper bounds."""
    if upper[0] <= lower[0] or upper[1] <= lower[1]:
        raise ValueError("box upper corner must be above/right of lower corner")
    X, Y = grid.cell_coords()
    return np.asarray(
        (lower[0] <= X) & (upper[0] >= X) & (lower[1] <= Y) & (upper[1] >= Y), dtype=bool
    )


def solid_face_masks(solid: BoolArray) -> tuple[BoolArray, BoolArray]:
    """Return boolean masks for u/v faces touching at least one solid cell.

    ``u`` has one extra column and ``v`` one extra row.  Boundary faces are included when a
    solid touches the domain boundary; interior solid faces are the union of their two
    adjacent cell masks.
    """
    solid = np.asarray(solid, dtype=bool)
    ny, nx = solid.shape
    um = np.zeros((ny, nx + 1), dtype=bool)
    vm = np.zeros((ny + 1, nx), dtype=bool)
    um[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
    um[:, 0] = solid[:, 0]
    um[:, -1] = solid[:, -1]
    vm[1:-1, :] = solid[:-1, :] | solid[1:, :]
    vm[0, :] = solid[0, :]
    vm[-1, :] = solid[-1, :]
    return um, vm


def apply_solid_velocity(
    u: Array,
    v: Array,
    solid: BoolArray | None,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> tuple[Array, Array]:
    """Prescribe obstacle velocity on all solid-touching faces, in place."""
    if solid is None:
        return u, v
    um, vm = solid_face_masks(np.asarray(solid, dtype=bool))
    u[um] = velocity[0]
    v[vm] = velocity[1]
    return u, v


@dataclass(frozen=True)
class MovingObstacle:
    """A translated circle/box whose mask can be evaluated at any time."""

    kind: str
    center0: tuple[float, float]
    size: float
    velocity: tuple[float, float] = (0.0, 0.0)

    def center(self, t: float) -> tuple[float, float]:
        return self.center0[0] + self.velocity[0] * t, self.center0[1] + self.velocity[1] * t

    def mask(self, grid: Grid, t: float) -> BoolArray:
        center = self.center(t)
        if self.kind == "circle":
            return circle_mask(grid, center, self.size)
        if self.kind == "box":
            half = 0.5 * self.size
            return box_mask(
                grid,
                (center[0] - half, center[1] - half),
                (center[0] + half, center[1] + half),
            )
        raise ValueError(f"unknown obstacle kind {self.kind!r}")

    def apply(self, u: Array, v: Array, grid: Grid, t: float) -> BoolArray:
        mask = self.mask(grid, t)
        apply_solid_velocity(u, v, mask, self.velocity)
        return mask
