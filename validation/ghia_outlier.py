"""Examine the Ghia et al. (1982) Re=400 datum ``v(x=.9063) = -.23827``.

The cavity validation excludes this single printed reference point from its primary error
metric and keeps it in the all-point metric.  This driver asks the question the exclusion
leaves open: is the datum an isolated outlier in the published table, or is the numerical
profile wrong in that region of the wall shear layer?

Four pieces of evidence are produced, all from runs in this repository.

1. **The published table itself.**  The transcribed columns (Re=100, 400 and 1000 are the
   three present in ``validation/data/ghia_1982.csv``) are examined without running anything.
   Along the moving-wall shear layer the ``v`` profile is unimodal, so between consecutive
   stations on the same flank the secant slopes must be of comparable size.  We report those
   secants and their ratio at the disputed station for every transcribed column.
2. **Reinterpolated reference.**  A monotone cubic (PCHIP) through the Re=400 column, built
   without the ``x=.9063`` point, gives an independent estimate at ``x=.9063``.
3. **A grid ladder at Re=400.**  ``v`` at ``(x=.9063, y=.5)`` at n=48, 64, 96 and 128 with the
   same solver settings as the milestone runs.  If the numerical value were the problem it
   would move towards the printed datum with refinement; if the datum is the problem, the
   numerical value converges on the reinterpolated reference instead.
4. **The same kink measure on the numerical profile**, so the reference column and the
   computed profile are judged by one yardstick.

Run ``python -m validation.ghia_outlier``; results go to ``outputs/m7/ghia``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from validation.lid_cavity import run_cavity

OUT = Path("outputs/m7/ghia")

#: The station under examination, as printed by Ghia et al. (1982).
STATION = 0.9063
STATION_VALUE = -0.23827

#: Interior stations of the moving-wall shear layer in Table II, ordered towards the wall.
SHEAR_LAYER = (0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688)

#: Reynolds numbers transcribed in this repository's copy of Table II.
TABLE_REYNOLDS = (100, 400, 1000)


def load_table(reynolds: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``v`` column for one Reynolds number, sorted by ``x``."""
    from validation.lid_cavity import reference

    x, v = reference(reynolds)["v_horizontal_centerline"]
    if x.size == 0:
        raise ValueError(f"no v_horizontal_centerline rows for Re={reynolds}")
    order = np.argsort(x)
    return np.asarray(x[order]), np.asarray(v[order])


def _secants(x: np.ndarray, v: np.ndarray, station: float = STATION) -> dict[str, float]:
    """Secant slopes immediately upstream and downstream of ``station``."""
    order = np.argsort(x)
    x, v = x[order], v[order]
    values = np.interp([0.8594, station, 0.9453], x, v)
    upstream = float((values[1] - values[0]) / (station - 0.8594))
    downstream = float((values[2] - values[1]) / (0.9453 - station))
    ratio = upstream / downstream if downstream != 0.0 else float("inf")
    # On a smooth unimodal profile the two secants differ only by the local curvature; a ratio
    # far from one means the middle point sits off its own curve.
    return {
        "secant_upstream": upstream,
        "secant_downstream": downstream,
        "slope_ratio": float(ratio),
        "kink": float(abs(np.log(abs(ratio)))) if ratio != 0.0 else float("inf"),
    }


def shear_layer_rows() -> list[dict[str, Any]]:
    """The moving-wall shear layers of every tabulated Reynolds number."""
    rows = []
    for reynolds in TABLE_REYNOLDS:
        x, v = load_table(reynolds)
        stations = []
        for x_station in SHEAR_LAYER:
            index = int(np.argmin(np.abs(x - x_station)))
            stations.append({"x": float(x[index]), "v": float(v[index])})
        peak = max(stations, key=lambda item: abs(item["v"]))
        rows.append(
            {
                "reynolds": reynolds,
                "stations": stations,
                "peak_station": peak["x"],
                **_secants(x, v),
            }
        )
    return rows


def reinterpolated_reference(reynolds: int = 400) -> dict[str, Any]:
    """Monotone-cubic estimate at the disputed station, built without that station."""
    x, v = load_table(reynolds)
    keep = ~np.isclose(x, STATION)
    curve = PchipInterpolator(x[keep], v[keep])
    estimate = float(curve(STATION))
    return {
        "reynolds": reynolds,
        "printed_value": STATION_VALUE,
        "reinterpolated_value": estimate,
        "difference": estimate - STATION_VALUE,
        "neighbours": [
            {"x": float(xi), "v": float(vi)}
            for xi, vi in zip(x[keep][-6:], v[keep][-6:], strict=True)
        ],
        "neighbour_secants": _secants(x, v),
    }


def _profile_at_station(row: dict[str, Any], station: float) -> float:
    x_line = np.asarray(row["x_line"])
    v_line = np.asarray(row["v_line"])
    order = np.argsort(x_line)
    return float(np.interp(station, x_line[order], v_line[order]))


def _numerical_secants(x_line: np.ndarray, v_line: np.ndarray) -> dict[str, float]:
    """The same secant measure applied to a computed profile."""
    return _secants(np.asarray(x_line), np.asarray(v_line))


#: The same solver settings as the milestone cavity runs.
def grid_ladder(
    ladder: list[int],
    reynolds: int = 400,
    max_time: float = 100.0,
    outdir: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for n in ladder:
        cache = None if outdir is None else outdir / f"reynolds{reynolds}_n{n}.json"
        if cache is not None and cache.exists() and not force:
            print(f"skipping n={n} (already measured; --force re-runs)", flush=True)
            rows.append(json.loads(cache.read_text()))
            continue
        print(f"running n={n}", flush=True)
        row = run_cavity(reynolds, n=n, max_time=max_time)
        value = _profile_at_station(row, STATION)
        rows.append(
            {
                "n": n,
                "reynolds": reynolds,
                "converged": bool(row["converged"]),
                "time": float(row["time"]),
                "steady_change": float(row["steady_change"]),
                "v_at_station": value,
                "error_against_printed": value - STATION_VALUE,
                "error_against_reinterpolated": None,
                "v_max_error": float(row["v_max_error"]),
                "v_max_error_all": float(row["v_max_error_all"]),
                "max_divergence": float(row["max_divergence"]),
                "wall_time": float(row["wall_time"]),
                **_numerical_secants(
                    np.asarray(row["x_line"]), np.asarray(row["v_line"])
                ),
                "x_line": np.asarray(row["x_line"]),
                "v_line": np.asarray(row["v_line"]),
            }
        )
        print(
            f"  n={n} v(.9063)={value:.6f} vs printed {STATION_VALUE:.5f}, "
            f"max div {row['max_divergence']:.2e}, converged={row['converged']}",
            flush=True,
        )
        if cache is not None:
            # Bank every finished grid level so an interrupted ladder resumes where it stopped.
            cache.write_text(json.dumps(_serial(rows[-1]), indent=2) + "\n")
    return rows


def _serial(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}


def _plot(outdir: Path, ladder: list[dict[str, Any]], estimate: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    for row in shear_layer_rows():
        x = [s["x"] for s in row["stations"]]
        v = [s["v"] for s in row["stations"]]
        ax.plot(x, v, "o-", ms=3, label=f"Re={row['reynolds']}")
    ax.axvline(STATION, color="black", ls=":", lw=1)
    ax.set(
        xlabel="x along y=.5",
        ylabel="v/U_lid",
        title="Table II moving-wall shear layer, transcribed Re columns",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    for row in ladder:
        ax.plot(row["x_line"], row["v_line"], label=f"n={row['n']}")
    printed_x, printed_v = load_table(400)
    ax.plot(printed_x, printed_v, "ko", ms=3, label="Ghia Re=400 (printed)")
    ax.plot(
        [STATION],
        [estimate["reinterpolated_value"]],
        "r*",
        ms=12,
        label=f"reinterpolated {estimate['reinterpolated_value']:.4f}",
    )
    ax.set(
        xlabel="x along y=.5",
        ylabel="v/U_lid",
        title="Re=400: grid ladder near the disputed datum",
        xlim=(0.75, 1.0),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    figure.savefig(outdir / "ghia_outlier.png", dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=OUT)
    parser.add_argument("--ladder", type=int, nargs="*", default=[48, 64, 96, 128])
    parser.add_argument("--max-time", type=float, default=100.0)
    parser.add_argument("--force", action="store_true", help="re-run completed grid levels")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    estimate = reinterpolated_reference()
    layers = shear_layer_rows()
    ladder = grid_ladder(
        args.ladder, max_time=args.max_time, outdir=args.outdir, force=args.force
    )
    for row in ladder:
        if "x_line" not in row:
            raise SystemExit(
                f"cached level n={row['n']} has no cached profile; re-run it with --force"
            )
    for row in ladder:
        row["error_against_reinterpolated"] = (
            row["v_at_station"] - estimate["reinterpolated_value"]
        )
    report = {
        "station": STATION,
        "printed_value": STATION_VALUE,
        "reinterpolation": estimate,
        "shear_layer": layers,
        "ladder": [_serial(row) for row in ladder],
    }
    (args.outdir / "ghia_outlier.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "n",
        "v_at_station",
        "error_against_printed",
        "error_against_reinterpolated",
        "converged",
        "steady_change",
        "v_max_error",
        "v_max_error_all",
        "max_divergence",
        "secant_upstream",
        "secant_downstream",
        "slope_ratio",
        "wall_time",
    ]
    with (args.outdir / "ghia_outlier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([_serial(row) for row in ladder])
    arrays: dict[str, Any] = {}
    for row in ladder:
        arrays[f"n{row['n']}_x_line"] = row["x_line"]
        arrays[f"n{row['n']}_v_line"] = row["v_line"]
    np.savez(args.outdir / "ghia_profiles.npz", **arrays)
    _plot(args.outdir, ladder, estimate)
    print(json.dumps({k: v for k, v in report.items() if k != "ladder"}, indent=2))
    print(json.dumps([_serial(row) for row in ladder], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
