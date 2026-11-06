"""M7 NumPy versus optional-Numba performance and equivalence benchmark.

Numba is not a base dependency and is never used implicitly by the solver.  The benchmark
warms up JIT compilation, checks transfer and short water-step results against NumPy, and
then reports wall times.  A transfer-only speedup is not presented as a whole-solver speedup:
pressure projection remains the same NumPy/SciPy code in both paths.

Run ``python -m validation.performance`` or ``mackerel benchmark``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig, _scatter, dam_break_particles
from mackerel.flip_numba import numba_available, scatter_numba
from mackerel.grid import Grid

OUT = Path("outputs/m7")


def _median_ms(call: Callable[[], Any], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def benchmark_transfer(n_values: tuple[int, ...] = (32, 64, 128), repeats: int = 7) -> list[dict[str, Any]]:
    """Benchmark both MAC components' particle transfer at several grid sizes."""
    rows: list[dict[str, Any]] = []
    for n in n_values:
        grid = Grid(n, n)
        particles = ParticleSet.from_region(
            grid, lambda X, Y: np.ones(X.shape, dtype=bool), particles_per_cell=4, seed=11
        )
        particles.u = np.sin(2.0 * np.pi * particles.x) + 0.1
        particles.v = np.cos(2.0 * np.pi * particles.y) - 0.2
        difference = 0.0
        for component, shape in (("u", grid.shape_u), ("v", grid.shape_v)):
            expected = _scatter(particles.u, particles.x, particles.y, grid.h, shape, component)
            actual = scatter_numba(particles.u, particles.x, particles.y, grid.h, shape, component)
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
            difference = max(difference, float(np.max(np.abs(expected[0] - actual[0]))),
                             float(np.max(np.abs(expected[1] - actual[1]))))
        # Benchmark the u transfer; the v kernel has the same loop structure and is checked
        # above.  Median times reduce the effect of a single scheduler interruption.
        numpy_ms = _median_ms(
            partial(_scatter, particles.u, particles.x, particles.y, grid.h, grid.shape_u, "u"),
            repeats,
        )
        numba_ms = _median_ms(
            partial(scatter_numba, particles.u, particles.x, particles.y, grid.h, grid.shape_u, "u"),
            repeats,
        )
        rows.append({
            "n": n, "particles": len(particles), "repeats": repeats,
            "numpy_ms": numpy_ms, "numba_ms": numba_ms,
            "speedup": numpy_ms / numba_ms, "max_abs_difference": difference,
        })
    return rows


def benchmark_water(n: int = 32, steps: int = 40) -> dict[str, Any]:
    """Compare complete short water steps, including the unchanged pressure solve."""
    def make(backend: str) -> FreeSurfaceWater:
        grid = Grid(n, 3 * n // 4)
        return FreeSurfaceWater(
            grid, dam_break_particles(grid),
            config=WaterConfig(dt=0.0005, transfer_backend=backend, pressure_tol=1e-8),
        )

    reference = make("numpy")
    accelerated = make("numba")
    # Compilation is warmed by benchmark_transfer in main; this call also makes the benchmark
    # safe to use directly.  Do not include a one-off JIT compilation in steady-state timing.
    scratch = make("numba")
    scratch.step()
    start = time.perf_counter()
    reference.run_steps(steps)
    numpy_seconds = time.perf_counter() - start
    start = time.perf_counter()
    accelerated.run_steps(steps)
    numba_seconds = time.perf_counter() - start
    errors = {
        name: float(np.max(np.abs(getattr(accelerated.particles, name) -
                                  getattr(reference.particles, name))))
        for name in ("x", "y", "u", "v")
    }
    return {
        "n": n, "steps": steps, "particles": len(reference.particles),
        "numpy_seconds": numpy_seconds, "numba_seconds": numba_seconds,
        "speedup": numpy_seconds / numba_seconds,
        "max_particle_error": max(errors.values()),
        "component_errors": errors,
        "numpy_final_divergence": reference.history[-1].divergence,
        "numba_final_divergence": accelerated.history[-1].divergence,
    }


def run_benchmark(repeats: int = 7, steps: int = 40) -> dict[str, Any]:
    """Return a serialisable benchmark report, or an explicit unavailable status."""
    report: dict[str, Any] = {
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numba_available": numba_available(),
        "numba_version": None,
        "compile_seconds": 0.0,
    }
    if not report["numba_available"]:
        report["status"] = "Numba is optional and is not installed; no accelerated timing claimed"
        return report
    import numba

    report["numba_version"] = numba.__version__
    # Force compilation before any timed sample and record its separate cost.
    grid = Grid(8, 8)
    p = dam_break_particles(grid)
    start = time.perf_counter()
    scatter_numba(p.u, p.x, p.y, grid.h, grid.shape_u, "u")
    report["compile_seconds"] = time.perf_counter() - start
    report["transfer"] = benchmark_transfer(repeats=repeats)
    report["water"] = benchmark_water(steps=steps)
    report["status"] = "equivalence checked; timings are steady-state medians"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = run_benchmark(repeats=args.repeats, steps=args.steps)
    (args.outdir / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
    if "transfer" in report:
        with (args.outdir / "performance_transfer.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report["transfer"][0]))
            writer.writeheader()
            writer.writerows(report["transfer"])
        water = report["water"]
        flat = {k: v for k, v in water.items() if k != "component_errors"}
        with (args.outdir / "performance_water.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = report["transfer"]
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        x = np.arange(len(rows))
        width = 0.36
        ax.bar(x - width / 2, [r["numpy_ms"] for r in rows], width, label="NumPy")
        ax.bar(x + width / 2, [r["numba_ms"] for r in rows], width, label="Numba")
        ax.set_xticks(x, [f"{r['n']}²\n{r['particles']} particles" for r in rows])
        ax.set_ylabel("median transfer time (ms)")
        ax.set_title("PIC/FLIP particle-to-grid transfer")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(args.outdir / "performance.png", dpi=150)
        plt.close(fig)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
