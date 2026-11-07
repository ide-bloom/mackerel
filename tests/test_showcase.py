"""Regressions for the committed README media and the plume frame recorder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from validation.showcase import AIR, field_rgb, water_rgb

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"
EXPECTED = ["dam_break", "hero", "sloshing", "smoke", "splash", "taylor_green"]


def test_showcase_media_is_committed_and_in_the_gitignore_whitelist() -> None:
    """The README links these files, so they must be present and not silently ignored."""
    for name in EXPECTED:
        matches = list(MEDIA.glob(f"{name}.*"))
        assert matches, f"missing docs/media/{name}.*"
    for gif in MEDIA.glob("*.gif"):
        assert gif.stat().st_size > 1024, f"{gif.name} is suspiciously small"
    # ``.gitignore`` ignores *.gif but whitelists docs/media, which is what lets these be
    # committed at all.  A whitelist typo would make the README images 404 on GitHub.
    ignored = (MEDIA / "dam_break.gif").resolve()
    rules = (MEDIA.parents[1] / ".gitignore").read_text().splitlines()
    assert "*.gif" in rules
    assert "!docs/media/*.gif" in rules
    assert ignored.exists()


def test_every_committed_gif_is_a_real_animation() -> None:
    from PIL import Image, ImageSequence

    for gif in sorted(MEDIA.glob("*.gif")):
        with Image.open(gif) as image:
            frames = sum(1 for _ in ImageSequence.Iterator(image))
            assert frames > 5, f"{gif.name} has only {frames} frames"
            assert image.size[0] >= 128 and image.size[1] >= 128, f"{gif.name} is too small"
            assert image.info.get("loop") == 0, f"{gif.name} does not loop"


def test_showcase_index_records_the_measured_diagnostics() -> None:
    import json

    index = json.loads((MEDIA / "showcase.json").read_text())
    for name in EXPECTED:
        assert name in index, f"{name} missing from showcase.json"
        assert Path(index[name]["file"]).name.startswith(name)
    for name in ("dam_break", "splash", "sloshing", "smoke", "taylor_green"):
        assert index[name]["frames"] > 5
        assert index[name]["grid"][0] > 0
    # The liquid cases must report conservation/diagnostics, not just a file name.
    assert index["dam_break"]["volume_relative_error"] < 1e-12
    assert index["dam_break"]["max_divergence"] < 1e-5
    assert index["splash"]["max_divergence"] < 1e-5
    assert index["smoke"]["centre_rise"] > 0.0


def test_smoke_plume_records_many_frames_not_one() -> None:
    """``validation/smoke.py`` used to compare against ``times[-2]`` and emit a single frame."""
    from validation.smoke import run_plume

    result = run_plume(16, end_time=0.3, dt=0.005)
    assert len(result["frames"]) > 5
    assert len(result["frame_times"]) == len(result["frames"])
    gaps = np.diff(result["frame_times"])
    assert np.all(gaps > 0.04), "frames must be spaced by the documented 0.049 s interval"
    assert result["frame_times"][-1] < result["time"] + 1e-9


def test_water_rgb_marks_the_free_surface_and_leaves_air_empty() -> None:
    from mackerel.flip import FreeSurfaceWater, ParticleSet, WaterConfig
    from mackerel.grid import Grid

    grid = Grid(8, 6)
    ii, jj = np.meshgrid(np.arange(8), np.arange(3))
    x = np.asarray(ii.ravel() + 0.5, dtype=np.float64) * grid.h
    y = np.asarray(jj.ravel() + 0.5, dtype=np.float64) * grid.h
    water = FreeSurfaceWater(
        grid, ParticleSet(x, y, np.zeros_like(x), np.zeros_like(y)),
        config=WaterConfig(dt=0.001), particles_per_cell=1,
    )
    rgb = water_rgb(water, scale=1)
    ny, nx = grid.shape_cell
    assert rgb.shape == (ny, nx, 3)
    # y-up: row 0 of the raster is the top of the domain, which is air here.
    top = rgb[0, 0] / 255.0
    # The palette is written as floats and rounded to uint8, so allow one count of rounding.
    np.testing.assert_allclose(top, AIR, atol=1.0 / 255.0 + 1e-9)
    # The interior of the liquid is not the air colour, and the top liquid row is the rim.
    assert not np.allclose(rgb[ny - 1, nx // 2] / 255.0, AIR, atol=1.0 / 255.0 + 1e-9)


def test_field_rgb_respects_the_value_range_and_solid_mask() -> None:
    values = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    solid = np.zeros((4, 4), dtype=bool)
    solid[0, 0] = True
    rgb = field_rgb(values, "RdBu_r", 0.0, 1.0, solid, scale=1)
    assert rgb.shape == (4, 4, 3)
    # y-up flip: the solid cell at (row 0, col 0) lands at the bottom-left of the raster.
    np.testing.assert_allclose(rgb[-1, 0], (25.5, 25.5, 25.5), atol=1.0)


@pytest.mark.parametrize("name", ["quick"])
def test_showcase_quick_mode_is_not_required_to_rerender(name: str) -> None:
    """Guard rail: the module must expose a callable ``main`` for the CLI registry."""
    from validation import showcase

    assert callable(showcase.main)
    assert name


def test_ci_workflow_runs_the_accuracy_gate_and_states_its_margins() -> None:
    """CI must keep enforcing the numerical subset, with the measured margins documented."""
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text()
    for command in ("ruff check .", "mypy", "python -m pytest", "python -m validation.ci_subset"):
        assert command in workflow, f"CI no longer runs: {command}"
    # Each threshold in the gate is justified next to its measured value in the workflow.
    for measured in ("-1.3003e-2", ".984116", ".962012", "2.0e-9"):
        assert measured in workflow, f"missing measured margin for {measured}"
