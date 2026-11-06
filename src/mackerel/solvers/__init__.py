"""Linear solvers for the pressure Poisson equation.

All four solvers share one interface: they act on a *flattened* problem description
consisting of the current guess, the right-hand side, the diagonal of the matrix and a
mask-free coefficient set, and they are all implemented **matrix-free** with NumPy.

The operator
------------
Discretising ``laplacian(p) = rhs`` with the 5-point stencil and the boundary conditions
described in :mod:`mackerel.project` gives a symmetric, negative-definite matrix (the
Laplacian has negative eigenvalues).  Every solver here works with ``A = -laplacian``, which
is symmetric positive definite, so the usual CG and multigrid theory applies directly.

Two things make this more than a textbook exercise:

* **Solid cells** (obstacles) are removed from the system. A pressure value in a solid cell
  is meaningless: the gradient there is never used. Instead the neighbour's pressure is
  mirrored, which is the discrete statement of ``dp/dn = 0`` at a solid wall. This is handled
  by the ``solid`` mask in :mod:`mackerel.project`.
* **Free-surface air** is a separate case. With ``free_surface=True`` the mask denotes air,
  and a mixed liquid-air face instead receives an exterior Dirichlet ``p=0`` diagonal term.
  Air is not silently treated as a Neumann solid.
* **The pure-Neumann null space.**  When every boundary is a wall or periodic, adding a
  constant to ``p`` changes nothing: the matrix is singular with a one-dimensional null
  space.  Jacobi and Gauss-Seidel drift in that direction, CG stalls or diverges.  Each
  solver here therefore projects the constant mode out of the correction on every iteration,
  which is exactly what makes the iteration converge to the *minimum-norm* solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import numpy as np

from mackerel.grid import Array

if TYPE_CHECKING:  # pragma: no cover
    from scipy.sparse import bsr_matrix, csc_matrix, csr_array, csr_matrix

    SparseBase = csr_array | csr_matrix | csc_matrix | bsr_matrix

    from mackerel.project import PoissonSystem

__all__ = [
    "SOLVERS",
    "ConjugateGradient",
    "GaussSeidel",
    "Jacobi",
    "Multigrid",
    "SolveResult",
    "Solver",
    "SolverName",
    "get_solver",
    "laplacian_apply",
]


# --------------------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------------------
def _sp() -> ModuleType:
    """Lazy ``scipy.sparse`` import (kept out of the module import path for speed)."""
    import scipy.sparse

    return scipy.sparse
def laplacian_apply(p: Array, system: PoissonSystem) -> Array:
    """Apply ``-laplacian`` (the SPD operator) to ``p``, honouring walls and solid cells.

    Interior: ``-div(grad p) = (4 p[i,j] - sum of neighbours) / h^2`` (with 3 neighbours at
    a non-periodic boundary, where the missing neighbour is mirrored: ``p[-1] = p[0]``).
    At a Dirichlet (open) boundary the neighbour value is 0 (the boundary pressure is fixed
    to zero) and the diagonal is correspondingly larger.
    """
    out = system.diag * p
    # x: face i couples cells i-1 and i, for i = 1 .. nx-1
    out[:, 1:] -= system.cx[:, 1:] * p[:, :-1]
    out[:, :-1] -= system.cx[:, 1:] * p[:, 1:]
    # y: face j couples cells j-1 and j, for j = 1 .. ny-1
    out[1:, :] -= system.cy[1:, :] * p[:-1, :]
    out[:-1, :] -= system.cy[1:, :] * p[1:, :]
    # faces 0 of each axis: a stored face that couples the two *opposite* edges.  It is a
    # periodic wrap (BC kind PERIODIC) or a straight wall (coefficient 0, no effect).
    out[:, 0] -= system.cx[:, 0] * p[:, -1]
    out[:, -1] -= system.cx[:, 0] * p[:, 0]
    out[0, :] -= system.cy[0, :] * p[-1, :]
    out[-1, :] -= system.cy[0, :] * p[0, :]
    # Only the *normal* differences belong in the Laplacian: the diagonal above must count
    # each coupled face exactly once.  It does (built in PoissonSystem.from_bc), so nothing
    # else to add here.
    return out / (system.h * system.h)


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------
@dataclass
class SolveResult:
    """Outcome of a linear solve."""

    p: Array
    iterations: int
    residual: float
    residual_history: list[float] = field(default_factory=list)


class Solver(Protocol):
    """Anything that can solve a :class:`~mackerel.project.PoissonSystem`."""

    name: str

    def solve(self, system: PoissonSystem, tol: float = 1e-10, maxiter: int = 500) -> SolveResult:
        ...


def _residual(p: Array, system: PoissonSystem) -> float:
    r = system.rhs - laplacian_apply(p, system)
    return float(np.max(np.abs(r)))


def _project_mean(p: Array, system: PoissonSystem) -> Array:
    """Remove the (weighted) mean of a field over the active cells -- the null space fix."""
    w = system.weight
    total = float(np.sum(w))
    if total <= 0.0:
        return p
    return p - float(np.sum(p * w)) / total


def _max_abs_divisor(system: PoissonSystem) -> float:
    """Scale for the relative residual: max |rhs|, floored to avoid a zero target."""
    scale = float(np.max(np.abs(system.rhs))) if system.rhs.size else 0.0
    return max(scale, 1e-30)


# --------------------------------------------------------------------------------------
# Jacobi
# --------------------------------------------------------------------------------------
@dataclass
class Jacobi:
    """Weighted Jacobi iteration.  Simple, provably convergent, and slow: the error of the
    highest-wavenumber modes shrinks fast, the error of the lowest-wavenumber modes barely
    at all -- ``O(n)`` iterations for an ``n x n`` grid."""

    omega: float = 1.0
    name: str = "jacobi"

    def solve(self, system: PoissonSystem, tol: float = 1e-10, maxiter: int = 500) -> SolveResult:
        p = system.guess.copy()
        h2 = system.h * system.h
        diag = system.diag / h2
        history: list[float] = []
        scale = _max_abs_divisor(system)
        for it in range(maxiter):
            if system.null_space:
                p = _project_mean(p, system)
            r = system.rhs - laplacian_apply(p, system)
            res = float(np.max(np.abs(r))) / scale if system.rhs.size else 0.0
            history.append(res)
            if res < tol or not np.isfinite(res):
                return SolveResult(p=p, iterations=it, residual=res, residual_history=history)
            p = p + self.omega * r / diag
        return SolveResult(p=p, iterations=maxiter, residual=history[-1], residual_history=history)


# --------------------------------------------------------------------------------------
# Gauss-Seidel (red-black)
# --------------------------------------------------------------------------------------
@dataclass
class GaussSeidel:
    """Red-black Gauss-Seidel: the grid is split like a checkerboard, each colour updated
    with the other's freshly written values.  Fully vectorised, and about twice as effective
    as Jacobi per sweep because information travels one cell per colour per sweep."""

    name: str = "gauss_seidel"

    def solve(self, system: PoissonSystem, tol: float = 1e-10, maxiter: int = 500) -> SolveResult:
        p = system.guess.copy()
        h2 = system.h * system.h
        diag = system.diag / h2
        ny, nx = p.shape
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        red = ((ii + jj) % 2 == 0)
        blue = ~red
        history: list[float] = []
        scale = _max_abs_divisor(system)
        for it in range(maxiter):
            if system.null_space:
                p = _project_mean(p, system)
            r = system.rhs - laplacian_apply(p, system)
            res = float(np.max(np.abs(r))) / scale if system.rhs.size else 0.0
            history.append(res)
            if res < tol or not np.isfinite(res):
                return SolveResult(p=p, iterations=it, residual=res, residual_history=history)
            for mask in (red, blue):
                # b = rhs + sum(coeff * neighbour)
                nb = np.zeros_like(p)
                nb[:, 1:] += system.cx[:, 1:] * p[:, :-1]
                nb[:, :-1] += system.cx[:, 1:] * p[:, 1:]
                nb[1:, :] += system.cy[1:, :] * p[:-1, :]
                nb[:-1, :] += system.cy[1:, :] * p[1:, :]
                b = system.rhs + nb / h2
                p[mask] = b[mask] / diag[mask]
        return SolveResult(p=p, iterations=maxiter, residual=history[-1], residual_history=history)


# --------------------------------------------------------------------------------------
# Conjugate Gradient
# --------------------------------------------------------------------------------------
@dataclass
class ConjugateGradient:
    """Matrix-free conjugate gradient, optionally diagonally preconditioned.

    CG is designed for symmetric positive definite systems, which is what the staggered
    Laplacian is; it converges in at most ``n`` iterations in exact arithmetic and in
    practice in ``O(sqrt(kappa))`` where ``kappa`` is the condition number -- for the 5-point
    Laplacian ``kappa = O(n^2)``, so ``O(n)`` iterations, like Jacobi but with a much smaller
    constant.  Preconditioning with the diagonal costs nothing and helps when the stencil is
    uneven (solid cells, Dirichlet rows).
    """

    precondition: bool = True
    name: str = "conjugate_gradient"

    def solve(self, system: PoissonSystem, tol: float = 1e-10, maxiter: int = 500) -> SolveResult:
        h2 = system.h * system.h
        diag = system.diag / h2
        x = system.guess.copy()
        if system.null_space:
            x = _project_mean(x, system)
        r = system.rhs - laplacian_apply(x, system)
        if system.null_space:
            r = _project_mean(r, system)
        z = r / diag if self.precondition else r.copy()
        d = z.copy()
        rz = float(np.sum(r * z))
        scale = _max_abs_divisor(system)
        history: list[float] = []
        for it in range(maxiter):
            res = float(np.max(np.abs(r))) / scale if system.rhs.size else 0.0
            history.append(res)
            if res < tol or not np.isfinite(res):
                return SolveResult(p=x, iterations=it, residual=res, residual_history=history)
            ad = laplacian_apply(d, system)
            denom = float(np.sum(d * ad))
            if denom == 0.0:
                return SolveResult(p=x, iterations=it, residual=res, residual_history=history)
            alpha = rz / denom
            x = x + alpha * d
            r = r - alpha * ad
            if system.null_space:
                x = _project_mean(x, system)
                r = _project_mean(r, system)
            z = r / diag if self.precondition else r
            rz_new = float(np.sum(r * z))
            if rz == 0.0:
                return SolveResult(p=x, iterations=it + 1, residual=res, residual_history=history)
            d = z + (rz_new / rz) * d
            rz = rz_new
        return SolveResult(p=x, iterations=maxiter, residual=history[-1], residual_history=history)


# --------------------------------------------------------------------------------------
# Multigrid (sparse Galerkin)
# --------------------------------------------------------------------------------------
def build_sparse(system: PoissonSystem) -> SparseBase:
    """Assemble the SPD operator ``-laplacian`` as a ``scipy.sparse`` CSR matrix.

    The matrix is *assembled from the same face coefficients* the matrix-free paths use, so
    the multigrid hierarchy is guaranteed to discretise the same operator the other solvers
    iterate on (a mismatch there is the classic source of a V-cycle that stalls).
    Solid cells keep their identity row (they are decoupled and their right-hand side is 0).
    """
    sp = _sp()
    ny, nx = system.solid.shape
    n = ny * nx
    idx = np.arange(n).reshape(ny, nx)
    rows: list[Array] = [idx.ravel()]
    cols: list[Array] = [idx.ravel()]
    vals: list[Array] = [(system.diag / (system.h * system.h)).ravel()]

    # x coupling: face i (i = 1..nx-1) joins (j, i-1) and (j, i); face 0 joins (j, nx-1)-(j, 0)
    cx = system.cx / (system.h * system.h)
    rows.append(idx[:, 1:].ravel())
    cols.append(idx[:, :-1].ravel())
    vals.append(-cx[:, 1:].ravel())
    rows.append(idx[:, :-1].ravel())
    cols.append(idx[:, 1:].ravel())
    vals.append(-cx[:, 1:].ravel())
    if np.any(cx[:, 0]):
        rows.append(idx[:, 0].ravel())
        cols.append(idx[:, -1].ravel())
        vals.append(-cx[:, 0].ravel())
        rows.append(idx[:, -1].ravel())
        cols.append(idx[:, 0].ravel())
        vals.append(-cx[:, 0].ravel())

    cy = system.cy / (system.h * system.h)
    rows.append(idx[1:, :].ravel())
    cols.append(idx[:-1, :].ravel())
    vals.append(-cy[1:, :].ravel())
    rows.append(idx[:-1, :].ravel())
    cols.append(idx[1:, :].ravel())
    vals.append(-cy[1:, :].ravel())
    if np.any(cy[0, :]):
        rows.append(idx[0, :].ravel())
        cols.append(idx[-1, :].ravel())
        vals.append(-cy[0, :].ravel())
        rows.append(idx[-1, :].ravel())
        cols.append(idx[0, :].ravel())
        vals.append(-cy[0, :].ravel())

    indices = (
        np.concatenate(cols).astype(np.int32),
        np.concatenate(rows).astype(np.int32),
    )
    out: SparseBase = sp.csr_matrix((np.concatenate(vals), indices), shape=(n, n))
    return out


def _interp_1d(n_fine: int, n_coarse: int) -> SparseBase:
    """Cell-centred 1-D interpolation matrix, fine <- coarse (rows sum to 1)."""
    sp = _sp()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for j in range(n_fine):
        # fine centre j is at physical (j+0.5)/n_fine = coarse coordinate (j+0.5)*nc/nf - 0.5
        u = (j + 0.5) * (n_coarse / n_fine) - 0.5
        u = min(max(u, 0.0), n_coarse - 1.0)
        k0 = int(np.floor(u))
        k1 = min(k0 + 1, n_coarse - 1)
        t = u - k0
        rows.extend([j, j])
        cols.extend([k0, k1])
        vals.extend([1.0 - t, t])
    m = sp.csr_matrix((vals, (rows, cols)), shape=(n_fine, n_coarse))
    sums = np.asarray(m.sum(axis=1)).ravel()
    out: SparseBase = sp.diags(1.0 / sums) @ m
    return out


def _prolongation_matrix(ny: int, nx: int, nyc: int, nxc: int) -> SparseBase:
    """2-D prolongation ``P`` (fine <- coarse) as a Kronecker product of 1-D operators."""
    sp = _sp()
    py = _interp_1d(ny, nyc)
    px = _interp_1d(nx, nxc)
    out: SparseBase = sp.kron(py, px, format="csr")
    return out


def _restriction_matrix(nyc: int, nxc: int, ny: int, nx: int) -> SparseBase:
    """Full-weighting restriction ``R``: the transpose of the prolongation, rows scaled to 1."""
    sp = _sp()
    p = _prolongation_matrix(ny, nx, nyc, nxc)
    rt = p.T.tocsr()
    sums = np.asarray(rt.sum(axis=1)).ravel()
    sums[sums == 0.0] = 1.0
    out: SparseBase = sp.diags(1.0 / sums) @ rt
    return out


@dataclass
class _Level:
    a: SparseBase      # sparse SPD operator, shape (n, n)
    p: SparseBase | None   # prolongation from the next coarser level (None on the finest)
    shape: tuple[int, int]


def _build_hierarchy(system: PoissonSystem, max_levels: int = 8) -> list[_Level]:
    """Build the sparse Galerkin hierarchy ``A_c = R A P`` down to a few cells per side."""
    levels = [_Level(a=build_sparse(system), p=None, shape=system.solid.shape)]
    while len(levels) < max_levels:
        ny, nx = levels[-1].shape
        if ny <= 4 or nx <= 4 or ny % 2 or nx % 2:
            break
        nyc, nxc = ny // 2, nx // 2
        prev = levels[-1]
        p = _prolongation_matrix(ny, nx, nyc, nxc)
        r = _restriction_matrix(nyc, nxc, ny, nx)
        a_coarse = (r @ prev.a @ p).tocsr()
        # ``p`` maps coarse coefficients to *fine* (this level's) coefficients, so it belongs
        # to the level being left, not to the coarse level being entered.
        prev.p = p
        levels.append(_Level(a=a_coarse, p=None, shape=(nyc, nxc)))
        if nyc <= 4 or nxc <= 4:
            break
    return levels


@dataclass
class Multigrid:
    """Geometric multigrid V-cycle with a sparse Galerkin coarse-grid hierarchy.

    One cycle is: weighted Jacobi pre-smoothing, restrict the *residual* to the next coarser
    grid, recurse, prolong the coarse correction back and add it, then post-smooth.  Smoothing
    removes the high-frequency part of the error quickly, and the coarse grid makes the
    low-frequency part look high-frequency, so one cycle removes all of it at once -- which is
    why the iteration count should be nearly independent of the grid size.

    The coarse operators are formed as ``A_c = R A P`` (Galerkin) from the same face
    coefficients the fine operator uses, with ``R`` the full-weighting transpose of the
    bilinear prolongation ``P``.  Building the coarse operator by *re-discretising* the
    equation on a grid of half the spacing instead -- the obvious shortcut -- leaves a
    coarse-grid truncation error that a V-cycle cannot remove; it was tried first here and
    stalled at a residual floor of 2.6e-3 relative (recorded in ``docs/milestones/M3.md``).
    """

    pre_smooth: int = 2
    post_smooth: int = 2
    omega: float = 0.8
    max_levels: int = 8
    name: str = "multigrid"
    _null_space: bool = False

    def _smooth(self, a: SparseBase, b: Array, x: Array, sweeps: int) -> Array:
        diag = np.asarray(a.diagonal(), dtype=np.float64)
        for _ in range(sweeps):
            x = x + self.omega * (b - a @ x) / diag
        return np.asarray(x, dtype=np.float64)

    def _vcycle(self, levels: list[_Level], level: int, b: Array, x: Array) -> Array:
        lv = levels[level]
        # NOTE: the termination case is the *last* level having no further coarse grid; a
        # level with ``p is None`` other than the finest would silently turn the V-cycle into
        # a direct solve (that bug made the first version of this class report "1 iteration"
        # for every grid, see docs/milestones/M3.md).
        if level == len(levels) - 1:
            # Coarsest level: the operator there is singular in the pure-Neumann case (the
            # constant mode), so a plain sparse LU emits MatrixRankWarning and can return
            # NaN.  Minimum-norm least squares (lsqr) is the numerically right tool and is
            # cheap at this size.
            import scipy.sparse.linalg as spla

            b_c = b - b.mean() if self._null_space else b
            sol = np.asarray(spla.lsqr(lv.a, b_c, atol=1e-14, btol=1e-14, iter_lim=400)[0])
            if self._null_space:
                sol = sol - sol.mean()
            return np.asarray(sol, dtype=np.float64)
        x = self._smooth(lv.a, b, x, self.pre_smooth)
        resid = b - lv.a @ x
        r = _restriction_matrix(*levels[level + 1].shape, *lv.shape)
        coarse_b = np.asarray(r @ resid, dtype=np.float64)
        coarse_x = self._vcycle(levels, level + 1, coarse_b, np.zeros_like(coarse_b))
        assert lv.p is not None
        x = x + np.asarray(lv.p @ coarse_x, dtype=np.float64)
        x = self._smooth(lv.a, b, x, self.post_smooth)
        return np.asarray(x, dtype=np.float64)

    def solve(self, system: PoissonSystem, tol: float = 1e-10, maxiter: int = 100) -> SolveResult:
        self._null_space = system.null_space
        levels = _build_hierarchy(system, self.max_levels)
        ny, nx = system.solid.shape
        b = system.rhs.ravel()
        x = system.guess.ravel().copy()
        scale = _max_abs_divisor(system)
        history: list[float] = []
        for it in range(maxiter):
            resid = b - levels[0].a @ x
            if system.null_space:
                resid = resid - resid.mean()
            res = float(np.max(np.abs(resid))) / scale if b.size else 0.0
            history.append(res)
            if res < tol or not np.isfinite(res):
                return SolveResult(
                    p=x.reshape(ny, nx), iterations=it, residual=res, residual_history=history
                )
            x = self._vcycle(levels, 0, b, x)
        return SolveResult(
            p=x.reshape(ny, nx), iterations=maxiter, residual=history[-1],
            residual_history=history,
        )


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------
class SolverName(str):
    """Marker type for solver names (kept a ``str`` so it works with argparse and reprs well)."""


SOLVERS: dict[str, type[Solver]] = {
    "jacobi": Jacobi,
    "gauss_seidel": GaussSeidel,
    "conjugate_gradient": ConjugateGradient,
    "multigrid": Multigrid,
}


def get_solver(name: str) -> Solver:
    """Instantiate a solver by name (``solvers.SOLVERS`` keys)."""
    try:
        cls = SOLVERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown solver {name!r}; choose from {sorted(SOLVERS)}") from exc
    instance: Solver = cls()
    return instance
