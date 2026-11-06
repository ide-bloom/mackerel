"""Task 6 regressions for conservative paired reseeding of the PIC/FLIP liquid."""

from __future__ import annotations

import numpy as np
import pytest

from mackerel.flip import (
    FreeSurfaceWater,
    ParticleSet,
    WaterConfig,
    dam_break_particles,
)
from mackerel.grid import Grid


def _slab_with_defects(grid: Grid, extra: int = 8) -> ParticleSet:
    """Four particles per cell over columns 2..9 and rows 1..7, with planted defects.

    Cell (5, 4) is emptied (an interior hole), cell (6, 6) keeps one particle (a starved
    interior cell) and cell (7, 4) receives ``extra`` particles (a donor).
    """
    gi, gj = np.meshgrid(np.arange(2, 10), np.arange(1, 8))
    ii, jj = gi.ravel(), gj.ravel()
    hole = (ii == 5) & (jj == 4)
    starved = (ii == 6) & (jj == 6)
    xs, ys = [], []
    for k in range(4):
        ox = (k % 2 + 0.5) / 2
        oy = (k // 2 + 0.5) / 2
        keep = ~hole & ~(starved & (k > 0))
        xs.append((ii[keep] + ox) * grid.h)
        ys.append((jj[keep] + oy) * grid.h)
    xs.append((7 + 0.15 + 0.05 * np.arange(extra)) * grid.h)
    ys.append((4 + 0.2 + 0.05 * np.arange(extra)) * grid.h)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return ParticleSet(x, y, np.full_like(x, 0.3), np.full_like(y, -0.2))


def _defect_water(extra: int = 8, **overrides: object) -> FreeSurfaceWater:
    grid = Grid(12, 9)
    options: dict[str, object] = {
        "dt": 0.001, "reseed": True, "min_particles_per_cell": 2,
        "max_particles_per_cell": 8, "reseed_radius": 5.0, "reseed_cooldown": 0,
    }
    options.update(overrides)
    config = WaterConfig(**options)  # type: ignore[arg-type]
    return FreeSurfaceWater(grid, _slab_with_defects(grid, extra), config=config)


def test_reseed_fills_interior_hole_and_starved_cell_without_creating_particles() -> None:
    water = _defect_water()
    before = water.interior_deficits()
    assert before == {"interior_holes": 1.0, "interior_starved": 1.0, "fringe_starved": 0.0}
    count = len(water.particles)
    velocity = water.particles.u.copy()
    record = water.reseed()
    after = water.interior_deficits()
    assert after["interior_holes"] == 0.0
    assert after["interior_starved"] == 0.0
    # The exchange is paired: the count, the velocities and therefore the nominal volume are
    # untouched, and only positions changed.
    assert len(water.particles) == count
    assert record["particle_count"] == float(count)
    np.testing.assert_array_equal(water.particles.u, velocity)
    assert record["moved"] == 3.0
    assert record["donor_cells"] == 1.0
    assert record["recipient_cells"] == 2.0
    assert record["after"]["cv"] < record["before"]["cv"]


def test_reseed_keeps_the_fringe_and_respects_the_radius() -> None:
    water = _defect_water()
    before = water.density_metrics()
    before_deficits = water.interior_deficits()
    record = water.reseed()
    assert water.interior_deficits()["fringe_starved"] == before_deficits["fringe_starved"] == 0.0
    assert before["over_target"] == 1.0
    assert record["max_move_distance"] <= 5.0 * water.grid.h
    # No particle may be placed outside the liquid: every recipient stays inside the domain.
    assert np.all(water.particles.x > 0.0)
    assert np.all(water.particles.x < water.grid.lx)
    assert np.all(water.particles.y > 0.0)
    assert np.all(water.particles.y < water.grid.ly)


def test_reseed_is_deterministic_for_a_fixed_seed_and_changes_with_it() -> None:
    first = _defect_water()
    second = _defect_water()
    first.reseed()
    second.reseed()
    np.testing.assert_array_equal(first.particles.x, second.particles.x)
    np.testing.assert_array_equal(first.particles.y, second.particles.y)
    other = _defect_water(reseed_seed=7)
    other.reseed()
    assert not np.array_equal(first.particles.x, other.particles.x)


def test_far_donor_is_not_used_when_no_particle_is_within_the_radius() -> None:
    water = _defect_water(reseed_radius=0.5)
    record = water.reseed()
    assert record["moved"] == 0.0
    assert record["unfilled_slots"] == 3.0
    assert water.interior_deficits()["interior_holes"] == 1.0


def test_reseeding_is_off_by_default_and_leaves_the_solution_unchanged() -> None:
    assert WaterConfig().reseed is False
    grid = Grid(24, 18)
    runs = []
    for reseed in (False, False):
        water = FreeSurfaceWater(
            grid, dam_break_particles(grid, width=0.25, height=0.5),
            config=WaterConfig(dt=0.0005, reseed=reseed),
        )
        water.run_steps(20)
        runs.append(water)
    assert runs[0].reseed_history == []
    np.testing.assert_array_equal(runs[0].particles.x, runs[1].particles.x)
    # A reseeded run of the same length must not be silently identical.
    reseeded = FreeSurfaceWater(
        grid, dam_break_particles(grid, width=0.25, height=0.5),
        config=WaterConfig(dt=0.0005, reseed=True, reseed_radius=5.0),
    )
    reseeded.run_steps(20)
    assert len(reseeded.reseed_history) == 20
    assert isinstance(reseeded.reseed_history, list)


def test_cooldown_stops_a_particle_being_relocated_on_consecutive_steps() -> None:
    water = _defect_water(reseed_cooldown=5)
    first = water.reseed()
    assert first["moved"] == 3.0
    moved_x = water.particles.x.copy()
    # A second immediate call has nothing left to do: the defects are filled and the donor is
    # at the nominal density, so no particle moves at all.
    second = water.reseed()
    assert second["moved"] == 0.0
    np.testing.assert_array_equal(water.particles.x, moved_x)
    assert water.interior_deficits()["interior_holes"] == 0.0


def test_reseed_improves_cell_density_on_a_short_dam_break() -> None:
    grid = Grid(48, 36)
    particles = dam_break_particles(grid, width=0.25, height=0.5)
    water = FreeSurfaceWater(
        grid, particles,
        config=WaterConfig(dt=0.0005, reseed=True, reseed_radius=5.0, reseed_cooldown=10),
    )
    initial = water.interior_deficits()
    assert initial == {"interior_holes": 0.0, "interior_starved": 0.0, "fringe_starved": 0.0}
    count = len(water.particles)
    volume = water.water_volume()
    water.run_steps(250)
    final = water.interior_deficits()
    assert final["interior_holes"] == 0.0
    assert final["interior_starved"] == 0.0
    assert len(water.particles) == count
    assert water.water_volume() == volume
    assert np.all(np.isfinite(water.particles.x))
    assert water.history[-1].divergence < 1e-6
    moved = sum(event["moved"] for event in water.reseed_history)
    assert 0 < moved < 0.25 * count


def test_hole_rule_rejects_an_empty_exterior_cell_with_three_liquid_neighbours() -> None:
    grid = Grid(10, 10)
    counts = np.zeros(grid.shape_cell, dtype=np.intp)
    counts[4:7, 4:7] = 4
    counts[5, 5] = 0  # only the fully enclosed cell is empty
    water = FreeSurfaceWater(
        grid, ParticleSet(np.empty(0), np.empty(0), np.empty(0), np.empty(0)),
        config=WaterConfig(dt=0.001, reseed=True),
    )
    recipients = water._reseed_recipients(counts)
    # Only the fully enclosed centre is a legitimate hole.  A three-neighbour rule would also
    # have accepted the four empty corners of the block, which are open to the exterior air.
    assert recipients.tolist() == [grid.nx * 5 + 5]
    # The four empty corners of the block each have three liquid neighbours and must not be
    # treated as holes, because they are open to the exterior air.
    for corner in (grid.nx * 4 + 4, grid.nx * 4 + 6, grid.nx * 6 + 4, grid.nx * 6 + 6):
        assert corner not in recipients.tolist()


def test_invalid_reseed_configuration_is_rejected() -> None:
    grid = Grid(12, 9)
    particles = _slab_with_defects(grid)
    with pytest.raises(ValueError):
        FreeSurfaceWater(grid, particles, config=WaterConfig(reseed=True, reseed_interval=0))
    with pytest.raises(ValueError):
        FreeSurfaceWater(
            grid, particles,
            config=WaterConfig(reseed=True, min_particles_per_cell=5, max_particles_per_cell=4),
        )
    with pytest.raises(ValueError):
        FreeSurfaceWater(
            grid, particles,
            config=WaterConfig(reseed=True, reseed_target=9, max_particles_per_cell=8),
        )
    with pytest.raises(ValueError):
        FreeSurfaceWater(grid, particles, config=WaterConfig(reseed=True, reseed_radius=0.0))
    with pytest.raises(ValueError):
        FreeSurfaceWater(grid, particles, config=WaterConfig(reseed=True, reseed_cooldown=-1))


def test_screen_exchange_is_conservative_and_fills_both_planted_defects() -> None:
    from validation.flip_reseed import _screen_exchange

    screen = _screen_exchange()
    assert screen["particles_before"] == screen["particles_after"] == 225.0
    assert screen["volume_before"] == screen["volume_after"]
    assert screen["moved"] == 3.0
    assert screen["max_move_distance_over_h"] <= 5.0
    assert screen["before"]["interior_holes"] == 1.0
    assert screen["before"]["interior_starved"] == 1.0
    assert screen["after"]["interior_holes"] == 0.0
    assert screen["after"]["interior_starved"] == 0.0
    assert screen["after"]["fringe_starved"] == screen["before"]["fringe_starved"]
    assert screen["after"]["rms_target_error"] < screen["before"]["rms_target_error"]
    assert screen["after"]["cv"] < screen["before"]["cv"]


def test_reseeded_dam_break_keeps_front_error_and_removes_interior_defects() -> None:
    """The documented 48x36 outcome, recomputed from the raw driver rather than a stored file."""
    from validation.flip import run_dam_break
    from validation.flip_reseed import CONFIG

    control = run_dam_break(n=48, reseed=False, reseed_config=CONFIG)
    reseeded = run_dam_break(n=48, reseed=True, reseed_config=CONFIG)
    assert control["reseed_summary"] is None
    summary = reseeded["reseed_summary"]
    assert summary is not None
    assert summary["moved"] > 0
    assert summary["max_move_distance_over_h"] <= CONFIG["reseed_radius"]
    assert reseeded["final_particles"] == control["final_particles"] == 1152
    assert reseeded["volume_relative_error"] == 0.0
    assert reseeded["final_deficits"]["interior_holes"] == 0.0
    assert reseeded["final_deficits"]["interior_starved"] == 0.0
    assert control["final_deficits"]["interior_holes"] + control["final_deficits"][
        "interior_starved"
    ] > 0
    # The measured effect is small: reseeding must not be able to pass itself off as a fix for
    # the M7 refinement failure documented for this benchmark.
    assert abs(reseeded["front_rmse"] - control["front_rmse"]) < 0.05 * control["front_rmse"]
