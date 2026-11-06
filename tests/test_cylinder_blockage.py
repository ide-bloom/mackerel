"""Smoke test for the parameterized cylinder blockage driver."""

from __future__ import annotations

import numpy as np
from validation.cylinder_blockage import run_case


def test_short_blockage_case_is_finite_and_records_geometry() -> None:
    row = run_case("short", nx=32, ny=16, lx=2.0, diameter=0.16, end_time=0.05)
    assert np.isclose(row["blockage"], 0.16)
    assert np.isclose(row["cells_per_diameter"], 2.56)
    assert np.isfinite(row["energy"])
    assert row["times"].size > 0
    assert row["max_divergence"] < 1e-6
