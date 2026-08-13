"""Tracker tests against synthetic ground-truth curves.

Curves are defined as voltage-vs-x functions in data space, converted to
pixel rows through the same axis convention the contract declares (higher
voltage = smaller pixel row), and sampled into per-column candidate dicts.
Assertions check both series-to-track identity and numerical closeness to
the known truth.
"""
import numpy as np
import pytest

from voltage_tracer.contract import validate_contract
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.tracker import (
    Track,
    fit_local_slope,
    initialize_tracks,
    match_cost,
    pixel_orientation,
    predict_next_y,
    track_columns,
)

# pixel frame: plot rows 20 (top, 4.5 V) .. 380 (bottom, 2.5 V)
V_TOP, V_BOT = 4.5, 2.5
PX_TOP, PX_BOT = 20.0, 380.0


def make_metrics(plot_w=400, plot_h=360, line_width=2.0, sep=35.0):
    """A FigureMetrics for the synthetic pixel frame (no real image), so
    the tracker derives the same thresholds it would on a ~400x360 figure."""
    return FigureMetrics(
        plot_w_px=plot_w,
        plot_h_px=plot_h,
        line_width_px=line_width,
        px_per_xunit=plot_w / 200.0,
        px_per_yunit=plot_h / (V_TOP - V_BOT),
        intercurve_color_sep=sep,
        text_regions=[],
    )


METRICS = make_metrics()


def v_to_py(v: float) -> float:
    return PX_BOT + (v - V_BOT) * (PX_TOP - PX_BOT) / (V_TOP - V_BOT)


def py_to_v(py: float) -> float:
    return V_BOT + (py - PX_BOT) * (V_TOP - V_BOT) / (PX_TOP - PX_BOT)


def make_contract(series_specs):
    return validate_contract(
        {
            "plot_bbox_px": [0, PX_TOP, 400, PX_BOT],
            "x_axis": {
                "scale": "linear",
                "ticks_px": [0, 400],
                "ticks_val": [0, 200],
            },
            "y_axis": {
                "scale": "linear",
                "ticks_px": [PX_BOT, PX_TOP],
                "ticks_val": [V_BOT, V_TOP],
            },
            "series": series_specs,
        }
    )


def columns_from_curves(curves: dict[str, callable], xs, merge_within=1.5):
    """Sample data-space curve functions into a candidate dict, merging
    candidates that rasterization couldn't distinguish (same as a real
    skeleton would produce one blob at a crossing)."""
    columns = {}
    for x in xs:
        ys = sorted(v_to_py(f(x)) for f in curves.values())
        merged = [[ys[0]]]
        for y in ys[1:]:
            if y - merged[-1][-1] <= merge_within:
                merged[-1].append(y)
            else:
                merged.append([y])
        columns[int(x)] = [float(np.mean(g)) for g in merged]
    return columns


def get_track(tracks, series_id):
    matches = [t for t in tracks if t.series_id == series_id]
    assert len(matches) == 1, f"expected 1 track for {series_id}, got {len(matches)}"
    return matches[0]


def assert_close_to_truth(track, curve_fn, tol_px=3.0, skip_head=0):
    pts = track.points[skip_head:]
    errs = [abs(y - v_to_py(curve_fn(x))) for x, y in pts]
    assert np.mean(errs) < tol_px, f"mean error {np.mean(errs):.2f}px"


# --- unit tests for the primitives ---


def test_fit_local_slope_recovers_line():
    t = Track("t", "s", "increasing", points_raw=[(x, 100.0 - 2.0 * x) for x in range(10)])
    assert fit_local_slope(t) == pytest.approx(-2.0)


def test_fit_local_slope_needs_two_points():
    t = Track("t", "s", "increasing", points_raw=[(0, 50.0)])
    assert fit_local_slope(t) == 0.0


def test_predict_next_y_extrapolates_across_gap():
    t = Track("t", "s", "increasing", points_raw=[(x, 100.0 - 2.0 * x) for x in range(10)])
    assert predict_next_y(t, 15) == pytest.approx(70.0)


def test_match_cost_penalizes_direction_violation():
    # pixel_sign -1: y must not exceed the extremum by more than tolerance
    ok = match_cost(50.0, 49.0, -1, extremum_y=50.0, reversal_tol_px=8.0)
    small_reversal = match_cost(50.0, 52.0, -1, extremum_y=50.0, reversal_tol_px=8.0)
    big_reversal = match_cost(50.0, 60.0, -1, extremum_y=50.0, reversal_tol_px=8.0)
    assert ok == pytest.approx(1.0)
    assert small_reversal == pytest.approx(2.0)  # within tolerance: no penalty
    assert big_reversal > 1e4


def test_match_cost_reversal_is_cumulative_not_per_step():
    """A slow drift against direction must be caught: even though each
    step is small, the distance from the extremum accumulates."""
    # extremum reached earlier at y=40; candidate creeps to 50 in 2px steps.
    # Each step is tiny, but 50 is 10px beyond the extremum: penalized.
    assert match_cost(48.0, 50.0, -1, extremum_y=40.0, reversal_tol_px=8.0) > 1e4
    # jitter around the extremum stays fine
    assert match_cost(42.0, 44.0, -1, extremum_y=40.0, reversal_tol_px=8.0) < 10


def test_pixel_orientation_standard_axis_is_negative():
    c = make_contract(
        [{"id": "s", "color_rgb": [255, 0, 0], "label": "x", "direction": "increasing"}]
    )
    assert pixel_orientation(c) == -1


# --- brief test case 1: single smooth increasing curve ---


def test_single_increasing_curve():
    charge = lambda x: 3.0 + 1.2 * x / 300 + 0.15 * np.sin(x / 60) * 0
    contract = make_contract(
        [{"id": "chg", "color_rgb": [255, 0, 0], "label": "0.5C charge",
          "direction": "increasing"}]
    )
    columns = columns_from_curves({"chg": charge}, range(0, 300))
    tracks = track_columns(columns, contract, METRICS)
    t = get_track(tracks, "chg")
    assert len(t.points) > 280
    assert_close_to_truth(t, charge, tol_px=2.0)
    ys = [y for _, y in t.points]
    assert ys[0] > ys[-1]  # pixel y falls as voltage rises


# --- brief test case 2: charge/discharge same color, crossing once ---


def test_single_crossing_keeps_identity():
    charge = lambda x: 3.1 + 1.2 * x / 300      # increasing, crosses discharge
    discharge = lambda x: 4.2 - 1.3 * x / 300   # decreasing
    contract = make_contract(
        [
            {"id": "chg", "color_rgb": [255, 0, 0], "label": "charge",
             "direction": "increasing"},
            {"id": "dis", "color_rgb": [255, 0, 0], "label": "discharge",
             "direction": "decreasing"},
        ]
    )
    columns = columns_from_curves({"c": charge, "d": discharge}, range(0, 300))
    tracks = track_columns(columns, contract, METRICS)
    t_chg = get_track(tracks, "chg")
    t_dis = get_track(tracks, "dis")
    # identity through the crossing: each track still follows its own curve
    assert_close_to_truth(t_chg, charge, tol_px=3.0)
    assert_close_to_truth(t_dis, discharge, tol_px=3.0)
    # and each maintained its direction (no bounce-back)
    chg_ys = np.array([y for _, y in t_chg.points])
    dis_ys = np.array([y for _, y in t_dis.points])
    assert (np.diff(chg_ys) <= 2.0).all(), "charge track reversed"
    assert (np.diff(dis_ys) >= -2.0).all(), "discharge track reversed"


# --- brief test case 3: two crossings, no identity swap on the second ---


def test_double_crossing_no_identity_swap():
    a = lambda x: 3.0 + 1.1 * (1 - np.exp(-x / 70))  # fast rise then flattens
    b = lambda x: 3.3 + 0.9 * x / 300                 # steady linear rise
    # sanity: these really do cross twice
    xs = np.arange(0, 300)
    sign_changes = np.sum(np.diff(np.sign([a(x) - b(x) for x in xs])) != 0)
    assert sign_changes == 2
    contract = make_contract(
        [
            {"id": "A", "color_rgb": [255, 0, 0], "label": "A",
             "direction": "increasing", "c_rate_rank": 2},
            {"id": "B", "color_rgb": [255, 0, 0], "label": "B",
             "direction": "increasing", "c_rate_rank": 1},
        ]
    )
    columns = columns_from_curves({"a": a, "b": b}, range(0, 300))
    tracks = track_columns(columns, contract, METRICS)
    # identity is defined by trajectory, so match tracks to curves by their
    # endpoints rather than by seed labels (same-direction seeds at x=0 are
    # inherently ambiguous; what matters is neither track swaps mid-run)
    finals = {t.series_id: t for t in tracks}
    assert len(finals) == 2
    by_end = sorted(tracks, key=lambda t: t.last_y)
    # at x=300: b(4.2V) is above a(4.08V) in voltage -> smaller pixel y
    assert_close_to_truth(by_end[0], b, tol_px=3.0, skip_head=10)
    assert_close_to_truth(by_end[1], a, tol_px=3.0, skip_head=10)


# --- brief test case 4: gap bridging ---


def test_gap_is_bridged_not_terminated():
    charge = lambda x: 3.0 + 1.4 * x / 300
    contract = make_contract(
        [{"id": "chg", "color_rgb": [255, 0, 0], "label": "charge",
          "direction": "increasing"}]
    )
    xs = [x for x in range(0, 300) if not (120 <= x < 140)]
    columns = columns_from_curves({"c": charge}, xs)
    tracks = track_columns(columns, contract, METRICS)
    assert len(tracks) == 1
    t = tracks[0]
    assert t.series_id == "chg"
    matched_xs = {x for x, _ in t.points}
    assert 110 in matched_xs and 150 in matched_xs, "track died at the gap"
    assert_close_to_truth(t, charge, tol_px=2.0)


# --- brief test case 5: multi-track proximity stress (Hungarian) ---


def test_four_discharge_curves_no_swaps():
    curves = {}
    specs = []
    for i in range(4):
        v0 = 4.35 - 0.09 * i           # staggered starts
        rate = 0.8 + 0.25 * i          # steeper drop for higher C-rate
        curves[f"d{i}"] = (lambda v0, rate: lambda x: v0 - rate * x / 300)(v0, rate)
        specs.append(
            {"id": f"d{i}", "color_rgb": [255, 0, 0], "label": f"rate{i}",
             "direction": "decreasing", "c_rate_rank": i + 1}
        )
    contract = make_contract(specs)
    columns = columns_from_curves(curves, range(0, 300))
    tracks = track_columns(columns, contract, METRICS)
    assert len(tracks) == 4
    for i in range(4):
        assert_close_to_truth(get_track(tracks, f"d{i}"), curves[f"d{i}"], tol_px=3.0)


# --- late-starting series spawns from the pending pool ---


def test_late_starting_series_is_spawned():
    early = lambda x: 4.2 - 1.0 * x / 300
    late = lambda x: 4.3 - 1.2 * (x - 150) / 300  # only exists from x=150
    contract = make_contract(
        [
            {"id": "early", "color_rgb": [255, 0, 0], "label": "early",
             "direction": "decreasing"},
            {"id": "late", "color_rgb": [255, 0, 0], "label": "late",
             "direction": "decreasing"},
        ]
    )
    columns = columns_from_curves({"e": early}, range(0, 300))
    late_cols = columns_from_curves({"l": late}, range(150, 300))
    for x, ys in late_cols.items():
        columns.setdefault(x, []).extend(ys)
    tracks = track_columns(columns, contract, METRICS)
    t_late = get_track(tracks, "late")
    assert t_late.points[0][0] <= 160, "late curve not picked up near its start"
    assert_close_to_truth(t_late, late, tol_px=3.0)
    assert_close_to_truth(get_track(tracks, "early"), early, tol_px=3.0)


# --- initialization heuristics ---


def test_initialize_seeds_by_expected_order():
    charge = lambda x: 3.0 + 1.2 * x / 300
    discharge = lambda x: 4.2 - 1.2 * x / 300
    contract = make_contract(
        [
            {"id": "chg", "color_rgb": [255, 0, 0], "label": "c",
             "direction": "increasing"},
            {"id": "dis", "color_rgb": [255, 0, 0], "label": "d",
             "direction": "decreasing"},
        ]
    )
    columns = columns_from_curves({"c": charge, "d": discharge}, range(0, 300))
    tracks, unseeded, _ = initialize_tracks(columns, contract, METRICS)
    assert not unseeded
    t_dis = next(t for t in tracks if t.series_id == "dis")
    t_chg = next(t for t in tracks if t.series_id == "chg")
    # discharge starts at high voltage -> top of plot -> smaller pixel y
    assert t_dis.points[0][1] < t_chg.points[0][1]


def test_track_does_not_wander_onto_sparse_noise_after_curve_ends():
    """After a curve's real end, scattered noise candidates (JPEG halos)
    must not extend the track -- its sparse tail gets trimmed."""
    charge = lambda x: 3.0 + 1.4 * x / 200
    contract = make_contract(
        [{"id": "chg", "color_rgb": [255, 0, 0], "label": "charge",
          "direction": "increasing"}]
    )
    columns = columns_from_curves({"c": charge}, range(0, 200))
    # sparse noise continuing the trend beyond the end (every 3rd column,
    # near where the prediction points): the worst case for coasting
    rng = np.random.default_rng(3)
    for x in range(200, 300, 3):
        y = v_to_py(charge(200)) - (x - 200) * 1.2 + rng.normal(0, 2)
        columns.setdefault(x, []).append(float(y))
    tracks = track_columns(columns, contract, METRICS)
    t = get_track(tracks, "chg")
    assert t.last_x <= 215, f"track wandered onto noise until x={t.last_x}"


def test_steep_single_curve_seeds_only_one_track():
    """Regression: a single steeply-descending curve at x-min must seed
    ONE track, not be split into two 'clusters' that seed both series of
    its color onto the same curve (the aqua-charge artifact)."""
    discharge = lambda x: 4.3 - 3.5 * x / 300  # steep: ~6.5px/column
    contract = make_contract(
        [
            {"id": "chg", "color_rgb": [255, 0, 0], "label": "c",
             "direction": "increasing"},
            {"id": "dis", "color_rgb": [255, 0, 0], "label": "d",
             "direction": "decreasing"},
        ]
    )
    columns = columns_from_curves({"d": discharge}, range(0, 100))
    tracks, unseeded, _ = initialize_tracks(columns, contract, METRICS)
    assert len(tracks) == 1
    assert tracks[0].series_id == "dis"  # slope-matched, not position-guessed
    assert [s.id for s in unseeded] == ["chg"]


def test_occluded_charge_head_no_identity_theft():
    """Regression for the real GraphEx2 failure: only the discharge curve
    is visible near x=0 (charge head occluded by the tangle). The
    discharge track must keep its own curve for its whole length, and the
    charge must spawn where its pixels appear -- not steal the discharge
    head and ride the wrong curve."""
    discharge = lambda x: 4.25 - 1.5 * x / 300
    charge = lambda x: 3.15 + 1.1 * x / 300
    contract = make_contract(
        [
            {"id": "chg", "color_rgb": [255, 0, 0], "label": "c",
             "direction": "increasing"},
            {"id": "dis", "color_rgb": [255, 0, 0], "label": "d",
             "direction": "decreasing"},
        ]
    )
    columns = columns_from_curves({"d": discharge}, range(0, 300))
    chg_cols = columns_from_curves({"c": charge}, range(25, 300))
    for x, ys in chg_cols.items():
        columns.setdefault(x, []).extend(ys)
    tracks = track_columns(columns, contract, METRICS)
    t_dis = get_track(tracks, "dis")
    t_chg = get_track(tracks, "chg")
    assert_close_to_truth(t_dis, discharge, tol_px=3.0)
    assert_close_to_truth(t_chg, charge, tol_px=3.0)
    assert t_chg.points[0][0] >= 20, "charge seeded onto occluded region"


def test_terminal_drop_is_extended():
    """A discharge's near-vertical final drop lives in the run extents,
    not the per-column means -- the extension pass must walk it down."""
    from voltage_tracer.tracker import extend_terminal_drops

    discharge = lambda x: 4.2 - 1.0 * x / 200
    contract = make_contract(
        [{"id": "dis", "color_rgb": [255, 0, 0], "label": "d",
          "direction": "decreasing"}]
    )
    columns = columns_from_curves({"d": discharge}, range(0, 200))
    tracks = track_columns(columns, contract, METRICS)
    t = get_track(tracks, "dis")
    plateau_end_y = t.last_y
    # runs dict: the drop is a tall fragmented run at x=200..201
    drop_bottom = v_to_py(V_BOT + 0.05)
    runs = {
        200: [(plateau_end_y + 4, plateau_end_y + 90, plateau_end_y + 47)],
        201: [(plateau_end_y + 95, drop_bottom, (plateau_end_y + 95 + drop_bottom) / 2)],
    }
    extend_terminal_drops(tracks, runs, METRICS)
    assert t.last_y == pytest.approx(drop_bottom, abs=1.0)
    # extension points descend monotonically
    ext_ys = [y for _, y in t.points[-10:]]
    assert all(b >= a for a, b in zip(ext_ys, ext_ys[1:]))


def test_terminal_extension_follows_slanted_fall():
    """A steep-but-slanted terminal fall spans many columns -- the
    extension window must slide along it, not stay fixed at the track
    end (the 20C-discharge case)."""
    from voltage_tracer.tracker import extend_terminal_drops

    t = Track("t", "s", "decreasing", points_raw=[(x, 100.0 + 0.5 * x) for x in range(40)],
              pixel_sign=1, y_extremum=119.5)
    # fall: 18px/column across columns 40..55
    runs = {x: [(119.5 + 18.0 * (x - 40), 119.5 + 18.0 * (x - 39), 0.0)]
            for x in range(40, 56)}
    extend_terminal_drops([t], runs, METRICS)
    assert t.last_y == pytest.approx(119.5 + 18.0 * 16, abs=2.0)
    assert t.last_x >= 54


def test_terminal_extension_is_smooth_on_sparse_columns():
    """Even when runs land on sparse columns, the densified extension must
    advance x smoothly (no vertical staircase stacked at one column)."""
    from voltage_tracer.tracker import extend_terminal_drops

    t = Track("t", "s", "decreasing",
              points_raw=[(x, 100.0 + 0.3 * x) for x in range(30)],
              pixel_sign=1, y_extremum=108.6)
    y0 = t.points[-1][1]
    # runs only every 4th column, each a tall fragment (the staircase trap)
    runs = {x: [(y0 + 6.0 * (x - 29), y0 + 6.0 * (x - 29) + 22.0, 0.0)]
            for x in range(29, 58, 4)}
    extend_terminal_drops([t], runs, METRICS)
    from collections import Counter
    ext = t.points[30:]
    counts = Counter(x for x, _ in ext)
    assert max(counts.values()) <= 4, f"staircase: {counts}"
    # x advances monotonically with y
    xs = [x for x, _ in ext]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_outlier_head_removed_but_ontrend_sparse_head_kept():
    from voltage_tracer.tracker import _trim_outlier_head

    # (a) stray leading point far off the following trend -> removed
    stray = Track("t", "s", "decreasing",
                  points_raw=[(0, 40.0)] + [(x, 100.0 + x) for x in range(5, 20)],
                  pixel_sign=1)
    _trim_outlier_head(stray)
    assert stray.points[0][0] == 5, "stray outlier head not removed"

    # (b) sparse seed head on-trend, then an occlusion gap -> kept
    ontrend = Track("t", "s", "decreasing",
                    points_raw=[(4, 100.0), (5, 101.5), (6, 103.0), (7, 104.5)]
                    + [(x, 100.0 + 1.5 * (x - 4)) for x in range(30, 45)],
                    pixel_sign=1)
    _trim_outlier_head(ontrend)
    assert ontrend.points[0] == (4, 100.0), "correct sparse seed head wrongly trimmed"


def test_terminal_extension_ignores_tiny_runs():
    from voltage_tracer.tracker import extend_terminal_drops

    t = Track("t", "s", "decreasing", points_raw=[(x, 100.0 + x) for x in range(50)],
              pixel_sign=1, y_extremum=149.0)
    before = list(t.points)
    runs = {50: [(150.0, 156.0, 153.0)]}  # only 7px of extension: noise
    extend_terminal_drops([t], runs, METRICS)
    assert t.points == before


def test_empty_columns_give_no_tracks():
    contract = make_contract(
        [{"id": "s", "color_rgb": [255, 0, 0], "label": "s",
          "direction": "increasing"}]
    )
    assert track_columns({}, contract, METRICS) == []


# --- bridging occlusion-split segments -------------------------------------

def _discharge_track():
    """A decreasing (discharge) track that stopped early at x=20, y=130."""
    ys = [100.0 + x * 1.5 for x in range(0, 21)]  # goes DOWN (pixel y grows)
    return Track("track_dis", "dis", "decreasing",
                 points_raw=[(x, 100.0 + x * 1.5) for x in range(0, 21)],
                 pixel_sign=1, y_extremum=max(ys))


def _seg2_columns(y0=160.0, slope=1.0, x0=60, x1=90):
    """A downstream decreasing chain (the recovered continuation), one
    candidate per column so _build_chains links it into a single chain."""
    return {x: [y0 + slope * (x - x0)] for x in range(x0, x1 + 1, 2)}


def _occluding_runs(x_lo, y_lo, x_hi, y_hi):
    """Other-colour runs sitting on the straight bridge path across the gap."""
    span = x_hi - x_lo
    return {
        x: [(y_lo + (y_hi - y_lo) * (x - x_lo) / span - 1.0,
             y_lo + (y_hi - y_lo) * (x - x_lo) / span + 1.0, 0.0)]
        for x in range(x_lo + 1, x_hi)
    }


def _bridge_contract():
    # one colour == one charge + one discharge (the structural invariant)
    return make_contract([
        {"id": "chg", "color_rgb": [230, 30, 30], "label": "c", "direction": "increasing"},
        {"id": "dis", "color_rgb": [230, 30, 30], "label": "d", "direction": "decreasing"},
    ])


def test_bridge_reconnects_across_occluded_gap():
    from voltage_tracer.tracker import bridge_occluded_segments
    t = _discharge_track()
    cols = _seg2_columns()
    other = _occluding_runs(20, 130.0, 60, 160.0)  # colour fills the gap
    bridge_occluded_segments([t], cols, other, _bridge_contract(), METRICS)
    assert t.last_x == 90, "continuation not stitched on"
    assert t.last_y == pytest.approx(190.0)
    # the curve stays monotone-decreasing (pixel y never goes back up)
    ys = [y for _, y in t.points_raw]
    assert all(b >= a - METRICS.reversal_tol_px for a, b in zip(ys, ys[1:]))


def test_no_bridge_when_gap_is_empty():
    from voltage_tracer.tracker import bridge_occluded_segments
    t = _discharge_track()
    bridge_occluded_segments([t], _seg2_columns(), {}, _bridge_contract(), METRICS)
    assert t.last_x == 20, "bridged across an empty gap (no occlusion evidence)"


def test_bridge_never_joins_opposite_direction():
    from voltage_tracer.tracker import bridge_occluded_segments
    t = _discharge_track()
    # a downstream INCREASING chain (would be the charge) must never attach
    up = {x: [200.0 - 1.0 * (x - 60)] for x in range(60, 91, 2)}
    other = _occluding_runs(20, 130.0, 60, 200.0)
    bridge_occluded_segments([t], up, other, _bridge_contract(), METRICS)
    assert t.last_x == 20, "joined an increasing chain onto a discharge"


def test_bridge_rejects_steep_join():
    from voltage_tracer.tracker import bridge_occluded_segments
    t = _discharge_track()
    # continuation starts far below -> the straight join would be near-vertical
    cols = _seg2_columns(y0=330.0)
    other = _occluding_runs(20, 130.0, 60, 330.0)
    bridge_occluded_segments([t], cols, other, _bridge_contract(), METRICS)
    assert t.last_x == 20, "accepted a steep (drop-like) join"
