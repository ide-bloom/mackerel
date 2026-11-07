"""Tests for the Ghia Re=400 outlier analysis."""

from __future__ import annotations

import numpy as np
from validation.ghia_outlier import (
    STATION,
    STATION_VALUE,
    load_table,
    reinterpolated_reference,
    shear_layer_rows,
)


def test_the_disputed_datum_is_present_verbatim_in_the_transcription() -> None:
    x, v = load_table(400)
    value = v[np.isclose(x, STATION)]
    assert value.size == 1
    assert float(value[0]) == STATION_VALUE


def test_the_reinterpolated_reference_disagrees_with_the_printed_datum() -> None:
    estimate = reinterpolated_reference()
    # The three-point monotone cubic through the neighbouring stations puts the datum about
    # a tenth of the lid speed away from the printed value; the sign matters more than the size.
    assert estimate["reinterpolated_value"] < -0.30
    assert estimate["difference"] < -0.10


def test_the_reference_column_has_a_large_secant_ratio_only_at_re_400() -> None:
    ratios = {row["reynolds"]: abs(row["slope_ratio"]) for row in shear_layer_rows()}
    assert ratios[400] > 5.0
    assert ratios[100] < 2.0
    assert ratios[1000] < 2.0


def test_every_transcribed_column_contains_the_shear_layer_stations() -> None:
    for row in shear_layer_rows():
        assert len(row["stations"]) == 8
        assert all(np.isfinite(station["v"]) for station in row["stations"])
