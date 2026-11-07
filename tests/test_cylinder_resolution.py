"""Tests for the windowed Strouhal estimator and its cylinder driver."""

from __future__ import annotations

import numpy as np
from validation.cylinder_resolution import run_case, windowed_strouhal


def _sine(frequency: float, end: float, dt: float = 0.025, **extra: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, end, dt)
    y = np.sin(2.0 * np.pi * frequency * t)
    for key, value in extra.items():
        if key == "drift":
            y = y + value * t
        elif key == "noise":
            y = y + value * np.random.default_rng(0).standard_normal(t.size)
    return t, y


def test_the_windowed_estimator_recovers_a_known_frequency() -> None:
    # St = f D / U with D=.16, U=1, so f=1.025 corresponds to the reference St=.164.
    t, y = _sine(1.025, 30.0, drift=0.01, noise=0.05)
    result = windowed_strouhal(t, y, diameter=0.16, speed=1.0, start=15.0)
    assert result["strouhal"] is not None
    assert abs(result["strouhal"] - 0.164) < 0.002


def test_the_window_and_cycle_count_describe_the_estimate() -> None:
    t, y = _sine(1.025, 30.0)
    result = windowed_strouhal(t, y, diameter=0.16, speed=1.0, start=15.0)
    # A 15-time-unit window at f=.1025 holds about 1.5 shedding cycles per unit... 15*1.025.
    assert np.isclose(result["window"], t[-1] - 15.0)
    assert np.isclose(result["cycles"], result["frequency"] * result["window"])
    assert np.isclose(result["frequency_resolution"], 1.0 / result["window"])
    # Both halves must agree when the signal really is a single tone.
    assert result["strouhal_half_spread"] < 0.002


def test_irregularly_sampled_times_are_resampled_before_windowing() -> None:
    # The driver records at "at least 0.025" intervals, so sample times are not exactly uniform.
    t, y = _sine(1.025, 30.0)
    t = t + np.random.default_rng(1).uniform(0.0, 0.01, t.size)
    result = windowed_strouhal(t, y, diameter=0.16, speed=1.0, start=15.0)
    assert result["strouhal"] is not None
    assert abs(result["strouhal"] - 0.164) < 0.002


def test_the_second_harmonic_does_not_masquerade_as_the_fundamental() -> None:
    t = np.arange(0.0, 30.0, 0.025)
    y = np.sin(2.0 * np.pi * 1.025 * t) + 0.6 * np.sin(2.0 * np.pi * 2.05 * t)
    result = windowed_strouhal(t, y, diameter=0.16, speed=1.0, start=15.0)
    assert result["strouhal"] is not None
    assert abs(result["strouhal"] - 0.164) < 0.004


def test_a_window_shorter_than_one_cycle_reports_its_cycle_count() -> None:
    # An under-resolved window is not silently rejected: the estimator states the window and
    # the number of shedding cycles in it, so a reader can judge the estimate.
    t, y = _sine(0.1, 5.0)
    result = windowed_strouhal(t, y, diameter=0.16, speed=1.0, start=0.0)
    assert result["strouhal"] is not None
    assert result["cycles"] < 1.0


def test_fewer_than_32_samples_gives_no_estimate_and_no_window() -> None:
    t, y = _sine(1.025, 0.4)
    result = windowed_strouhal(t, y, diameter=0.16, speed=1.0, start=0.0)
    assert result["strouhal"] is None
    assert result["window"] is None


def test_short_resolution_case_records_its_geometry_and_bookkeeping() -> None:
    row = run_case(
        "short",
        nx=32,
        ny=16,
        lx=2.0,
        diameter=0.16,
        end_time=0.05,
        dt=0.01,
    )
    assert np.isclose(row["blockage"], 0.16)
    assert np.isclose(row["cells_per_diameter"], 2.56)
    assert np.isclose(row["diameter_per_dt"], 16.0)
    assert np.isclose(row["cfl_advective"], 0.01 * 1.0 / row["spacing"])
    assert row["pressure_cap_hits"] == 0
    assert row["max_divergence"] < 1e-6
    assert row["times"].size > 0
