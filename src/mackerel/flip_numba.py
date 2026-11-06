"""Optional Numba kernel for the M7 particle-to-grid transfer.

The solver remains pure NumPy by default.  This module imports Numba only when the
``transfer_backend='numba'`` path is selected, so the base installation has no Numba
requirement.  Accumulation is deliberately serial: parallel ``prange`` updates would race on
a shared MAC face and would also make the result dependent on thread scheduling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mackerel.grid import Array

__all__ = ["numba_available", "scatter_numba"]


def numba_available() -> bool:
    """Return whether the optional Numba dependency can be imported."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _scatter_loop(
    values: Array, x: Array, y: Array, h: float, component: int,
    ny: int, nx: int,
) -> tuple[Array, Array]:
    out = np.zeros((ny, nx), dtype=np.float64)
    weight = np.zeros((ny, nx), dtype=np.float64)
    for k in range(values.size):
        if component == 0:  # u faces: x=i*h, y=(j+1/2)*h
            fx = x[k] / h
            fy = y[k] / h - 0.5
        elif component == 1:  # v faces: x=(i+1/2)*h, y=j*h
            fx = x[k] / h - 0.5
            fy = y[k] / h
        else:  # cell centres, useful for kernel-level tests
            fx = x[k] / h - 0.5
            fy = y[k] / h - 0.5
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        i0 = min(max(i0, 0), nx - 2)
        j0 = min(max(j0, 0), ny - 2)
        tx = min(max(fx - i0, 0.0), 1.0)
        ty = min(max(fy - j0, 0.0), 1.0)
        wx = 1.0 - tx
        wy = 1.0 - ty
        value = values[k]
        out[j0, i0] += wx * wy * value
        weight[j0, i0] += wx * wy
        out[j0, i0 + 1] += tx * wy * value
        weight[j0, i0 + 1] += tx * wy
        out[j0 + 1, i0] += wx * ty * value
        weight[j0 + 1, i0] += wx * ty
        out[j0 + 1, i0 + 1] += tx * ty * value
        weight[j0 + 1, i0 + 1] += tx * ty
    return out, weight


_COMPILED: Any = None


def scatter_numba(
    values: Array, x: Array, y: Array, h: float, shape: tuple[int, int], component: str
) -> tuple[Array, Array]:
    """Compile on first use and apply the same four-corner transfer as ``flip._scatter``."""
    global _COMPILED
    if _COMPILED is None:
        from numba import njit

        _COMPILED = njit(cache=True)(_scatter_loop)
    code = {"u": 0, "v": 1, "cell": 2}[component]
    ny, nx = shape
    result: tuple[Array, Array] = _COMPILED(values, x, y, h, code, ny, nx)
    return result
