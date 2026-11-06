"""Short Milestone 6 checks; long wake/plume runs live in validation/."""

from __future__ import annotations

import numpy as np
from validation.cylinder import strouhal_from_signal
from validation.moving import run_moving
from validation.smoke import run_plume


def test_strouhal_estimator_recovers_a_known_sine_frequency() -> None:
    t = np.arange(0.0, 20.0, 0.02)
    signal = np.sin(2.0 * np.pi * 1.25 * t)
    result = strouhal_from_signal(t, signal, diameter=0.16, speed=1.0, start=0.0)
    assert abs(result["strouhal"] - 0.2) < 0.002
    assert abs(result["fft_strouhal"] - 0.2) < 0.002


def test_short_plume_has_positive_buoyant_transport() -> None:
    result = run_plume(n=16, end_time=0.05, dt=0.005)
    assert result["final_centre_y"] > result["initial_centre_y"]
    assert result["max_temperature"] <= 1.0 + 1e-8
    assert result["min_temperature"] >= -1e-8
    assert result["final_divergence"] < 1e-6


def test_short_moving_obstacle_preserves_translation_and_face_velocity() -> None:
    result = run_moving(n=12, end_time=0.05, dt=0.005)
    assert abs(result["obstacle_final_x"] - result["expected_final_x"]) < 1e-12
    assert result["solid_face_velocity_error"] == 0.0
    assert result["max_divergence"] < 1e-6
