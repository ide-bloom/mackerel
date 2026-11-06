# mackerel

A small, executable fluid-simulation lab built milestone by milestone. The reference solvers
use NumPy on a staggered MAC grid; optional Numba acceleration is available for the M7
particle-transfer kernel.

## Quick start

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
mypy
```

Run a validated case or a small scene from the repository root:

```bash
python -m validation.flip          # PIC/FLIP dam break, splash and sloshing
python -m validation.performance   # optional NumPy/Numba benchmark
mackerel run cylinder --n 32 --steps 200
mackerel run dam_break --n 48 --steps 200
mackerel run splash --n 40 --steps 200
mackerel run sloshing --n 40 --steps 200
```

Generated figures, GIFs, CSV/NPZ data and JSON summaries are written below `outputs/` (which
is intentionally gitignored). MP4 output is used when `ffmpeg` is installed; otherwise the
rendering utilities retain GIF output.

## Milestone reports

* [M1 — MAC grid and diagnostics](docs/milestones/M1.md)
* [M2 — Advection schemes](docs/milestones/M2.md)
* [M3 — Pressure projection and iterative solvers](docs/milestones/M3.md)
* [M4 — Diffusion and the split solver](docs/milestones/M4.md)
* [M5 — Lid-driven cavity validation](docs/milestones/M5.md)
* [M6 — Obstacles, cylinder wake and plume demos](docs/milestones/M6.md)
* [M7 — PIC/FLIP free-surface water](docs/milestones/M7.md)

The mathematical conventions and design decisions are collected in [docs/MATH.md](docs/MATH.md)
and [docs/DESIGN.md](docs/DESIGN.md). Validation reference data and its provenance live in
`validation/data/`.
