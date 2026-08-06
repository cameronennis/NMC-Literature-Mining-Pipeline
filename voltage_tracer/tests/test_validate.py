import numpy as np

from voltage_tracer.contract import validate_contract
from voltage_tracer.defaults import DEFAULTS
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.tracker import Track
from voltage_tracer.validate import (
    VIOLATION,
    WARNING,
    check_capacity_ordering,
    check_monotonicity,
    check_smoothing_deviation,
    check_terminal_drop_sanity,
    validate_all,
)


def _metrics(line_width=2.0, w=500, h=400, text_regions=None):
    return FigureMetrics(
        plot_w_px=w, plot_h_px=h, line_width_px=line_width,
        px_per_xunit=1.5, px_per_yunit=100.0, intercurve_color_sep=40.0,
        text_regions=text_regions or [], d=DEFAULTS,
    )


def _track(ys, sign, series_id="s", xs=None):
    xs = xs if xs is not None else list(range(len(ys)))
    t = Track(series_id, series_id, "increasing" if sign > 0 else "decreasing",
              points_raw=[(int(x), float(y)) for x, y in zip(xs, ys)], pixel_sign=sign)
    return t


# --- monotonicity ---


def test_monotone_curve_passes():
    ys = 100 + 0.5 * np.arange(200)  # clean increasing (discharge, sign +1)
    assert check_monotonicity(_track(ys, +1), _metrics()) == []


def test_sub_stroke_jitter_not_flagged():
    rng = np.random.default_rng(0)
    ys = 100 + 0.5 * np.arange(200) + rng.normal(0, 0.6, 200)  # < stroke jitter
    findings = check_monotonicity(_track(ys, +1), _metrics(line_width=2.0))
    assert findings == []


def test_sustained_reversal_flagged():
    ys = list(100 + 0.5 * np.arange(100))
    # a sustained backward run: voltage-pixel goes DOWN for 10 pts (against +1)
    ys += list(ys[-1] - 1.5 * np.arange(1, 11))
    ys += list(ys[-1] + 0.5 * np.arange(1, 50))
    f = check_monotonicity(_track(ys, +1), _metrics())
    assert len(f) == 1 and f[0].severity == VIOLATION
    assert f[0].check == "monotonicity" and f[0].x_range is not None


def test_plateau_not_flagged_as_reversal():
    ys = list(100 + 0.5 * np.arange(80)) + [140.0] * 40 + list(140 + 0.5 * np.arange(40))
    assert check_monotonicity(_track(ys, +1), _metrics()) == []


# --- capacity ordering ---


def _contract_ranked():
    return validate_contract({
        "plot_bbox_px": [0, 0, 500, 400],
        "x_axis": {"scale": "linear", "ticks_px": [0, 500], "ticks_val": [0, 300]},
        "y_axis": {"scale": "linear", "ticks_px": [400, 0], "ticks_val": [2.5, 4.5]},
        "series": [
            {"id": "lo", "color_rgb": [255, 0, 0], "label": "0.1C d",
             "direction": "decreasing", "c_rate_rank": 1},
            {"id": "hi", "color_rgb": [0, 0, 255], "label": "5C d",
             "direction": "decreasing", "c_rate_rank": 2},
        ],
    })


def test_capacity_ordering_ok_when_consistent():
    lo = _track([100, 100], +1, "lo"); lo.data_raw = [(0.0, 4.0), (200.0, 3.0)]
    hi = _track([100, 100], +1, "hi"); hi.data_raw = [(0.0, 4.0), (60.0, 3.0)]
    assert check_capacity_ordering([lo, hi], _contract_ranked(), _metrics()) == []


def test_capacity_ordering_warns_on_inversion():
    # low-rate curve wrongly has LESS capacity than the high-rate one
    lo = _track([100, 100], +1, "lo"); lo.data_raw = [(0.0, 4.0), (50.0, 3.0)]
    hi = _track([100, 100], +1, "hi"); hi.data_raw = [(0.0, 4.0), (200.0, 3.0)]
    f = check_capacity_ordering([lo, hi], _contract_ranked(), _metrics())
    assert len(f) == 1 and f[0].severity == WARNING
    assert f[0].check == "capacity_ordering"


# --- terminal drop sanity ---


def test_trace_outside_plot_box_flagged():
    ys = list(range(390, 410))  # runs off the bottom (h=400)
    f = check_terminal_drop_sanity(_track(ys, +1), _metrics(h=400), _contract_ranked())
    assert any(x.check == "terminal_drop_sanity" and x.severity == VIOLATION for x in f)


def test_trace_into_text_region_flagged():
    t = _track([50] * 10, +1, xs=list(range(440, 450)))  # sits in the text band
    f = check_terminal_drop_sanity(
        t, _metrics(text_regions=[(430, 40, 470, 60)]), _contract_ranked())
    assert any(x.check == "terminal_drop_sanity" for x in f)


def test_clean_trace_no_terminal_flag():
    t = _track([100 + i for i in range(50)], +1, xs=list(range(10, 60)))
    assert check_terminal_drop_sanity(t, _metrics(), _contract_ranked()) == []


# --- smoothing deviation ---


def test_smoothing_deviation_flags_large_departure():
    xs = np.arange(100)
    raw = 100 + 0.5 * xs
    t = _track(raw, +1)
    sm = raw.copy()
    sm[40:55] -= 15.0  # smoother pulled a whole span far from raw
    t.points_smooth = [(int(x), float(y)) for x, y in zip(xs, sm)]
    f = check_smoothing_deviation(t, _metrics())
    assert len(f) == 1 and f[0].severity == WARNING
    assert f[0].check == "smoothing_deviation"


def test_faithful_smoothing_not_flagged():
    rng = np.random.default_rng(1)
    xs = np.arange(150)
    raw = 100 + 0.5 * xs + rng.normal(0, 1.0, 150)
    t = _track(raw, +1)
    # smoothed = gently denoised (small deviations only)
    sm = 100 + 0.5 * xs
    t.points_smooth = [(int(x), float(y)) for x, y in zip(xs, sm)]
    assert check_smoothing_deviation(t, _metrics()) == []


# --- aggregate ---


def test_validate_all_never_raises_and_aggregates():
    lo = _track([100, 100], +1, "lo"); lo.data_raw = [(0.0, 4.0), (50.0, 3.0)]
    lo.points_smooth = list(lo.points_raw)
    hi = _track([100, 100], +1, "hi"); hi.data_raw = [(0.0, 4.0), (200.0, 3.0)]
    hi.points_smooth = list(hi.points_raw)
    findings = validate_all([lo, hi], _contract_ranked(), _metrics())
    assert isinstance(findings, list)
    assert any(f.check == "capacity_ordering" for f in findings)
