import numpy as np
import pytest

from voltage_tracer.contract import validate_contract
from voltage_tracer.defaults import DEFAULTS
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.smooth import compute_point_weights, smooth_all, smooth_track
from voltage_tracer.tracker import PointInfo, Track


def _metrics(line_width=2.0, w=500, h=400):
    return FigureMetrics(
        plot_w_px=w, plot_h_px=h, line_width_px=line_width,
        px_per_xunit=1.5, px_per_yunit=100.0, intercurve_color_sep=40.0,
        text_regions=[], d=DEFAULTS,
    )


def _noisy_track(fn, xs, sign, noise=1.5, seed=0, cost=0.0, info=None):
    rng = np.random.default_rng(seed)
    truth = np.array([fn(x) for x in xs], dtype=float)
    ys = truth + rng.normal(0, noise, size=len(xs))
    t = Track("t", "s", "increasing" if sign > 0 else "decreasing",
              points_raw=[(int(x), float(y)) for x, y in zip(xs, ys)],
              pixel_sign=sign)
    if info is not None:
        t.info_raw = info
    else:
        t.info_raw = [PointInfo(True, 2, cost) for _ in xs]
    return t, truth


# --- denoising quality ---


def test_smoothed_is_closer_to_truth_than_raw():
    xs = np.arange(0, 200)
    fn = lambda x: 100.0 + 0.4 * x  # smooth monotone (pixel y increasing)
    t, truth = _noisy_track(fn, xs, sign=+1, noise=2.0, seed=1)
    smooth_track(t, _metrics(), active_track_count=1)
    raw = np.array([p[1] for p in t.points_raw])
    sm = np.array([p[1] for p in t.points_smooth])
    assert np.mean((sm - truth) ** 2) < np.mean((raw - truth) ** 2)
    # and within a few percent of the truth's range
    assert np.mean(np.abs(sm - truth)) < 0.03 * (truth.max() - truth.min())


def test_endpoints_pinned_exactly():
    xs = np.arange(0, 120)
    t, _ = _noisy_track(lambda x: 80.0 + 0.5 * x, xs, sign=+1, noise=2.0, seed=2)
    smooth_track(t, _metrics(), 1)
    # the start pins to the raw endpoint exactly
    assert t.points_smooth[0] == (t.points_raw[0][0], t.points_raw[0][1])
    # the end pins to the raw endpoint too, unless doing so would dip against
    # the declared (increasing) direction -- then it holds the monotone value
    x_last, y_last_raw = t.points_raw[-1]
    assert t.points_smooth[-1][0] == x_last
    assert t.points_smooth[-1][1] == pytest.approx(max(y_last_raw, t.points_smooth[-2][1]))


def test_non_destructive_raw_unchanged():
    xs = np.arange(0, 100)
    t, _ = _noisy_track(lambda x: 50.0 + 0.6 * x, xs, sign=+1, seed=3)
    before = list(t.points_raw)
    before_info = list(t.info_raw)
    smooth_track(t, _metrics(), 1)
    assert t.points_raw == before
    assert t.info_raw == before_info
    assert t.points_smooth is not None and t.points_smooth is not t.points_raw


def test_monotonicity_in_declared_direction():
    xs = np.arange(0, 150)
    t, _ = _noisy_track(lambda x: 60.0 + 0.7 * x, xs, sign=+1, noise=3.0, seed=4)
    smooth_track(t, _metrics(), 1)
    ys = np.array([p[1] for p in t.points_smooth])
    assert (np.diff(ys) >= -1e-9).all(), "pixel y must be non-decreasing (discharge)"

    t2, _ = _noisy_track(lambda x: 300.0 - 0.7 * x, xs, sign=-1, noise=3.0, seed=5)
    smooth_track(t2, _metrics(), 1)
    ys2 = np.array([p[1] for p in t2.points_smooth])
    assert (np.diff(ys2) <= 1e-9).all(), "pixel y must be non-increasing (charge)"


# --- sharp terminal drop preserved under the single uniform pass ---


def test_terminal_drop_corner_preserved():
    # a flat plateau then a steep drop; drop points are high-confidence
    xs = list(range(0, 100)) + list(range(100, 112))
    ys = [100.0] * 100 + [100.0 + 12 * k for k in range(1, 13)]
    info = [PointInfo(True, 2, 0.0) for _ in xs]  # all high-confidence
    t = Track("t", "s", "decreasing", pixel_sign=+1,
              points_raw=[(int(x), float(y)) for x, y in zip(xs, ys)], info_raw=info)
    smooth_track(t, _metrics(), 1)
    sm = np.array([p[1] for p in t.points_smooth])
    # the cutoff depth (final y) is not pulled inward beyond a stroke width
    assert abs(sm[-1] - ys[-1]) <= _metrics().line_width_px
    # the drop stays steep: last-segment slope close to raw
    assert sm[-1] - sm[-13] > 100.0


def test_terminal_dip_does_not_truncate_peak():
    # a charge (pixel y decreasing = voltage rising) that climbs to a peak,
    # then the final raw point dips slightly back -- a small terminal hook.
    # The smoothed tail must still reach the true peak, not be clamped down to
    # the non-extreme endpoint value. (Regression: P18 fig2a Li3InCl6/Li3PS4
    # charge, where the smoothed curve flatlined over the last ~6%.)
    ys = [300.0 - 1.7 * k for k in range(118)]  # pixel y falls to ~101 (peak)
    ys += [112.0, 130.0]                         # final points hook back up
    xs = list(range(len(ys)))
    info = [PointInfo(True, 2, 0.0) for _ in xs]  # ordinary confidence
    t = Track("t", "s", "increasing", pixel_sign=-1,
              points_raw=[(int(x), float(y)) for x, y in zip(xs, ys)], info_raw=info)
    smooth_track(t, _metrics(), 1)
    sm = np.array([p[1] for p in t.points_smooth])
    true_peak = min(ys)   # smallest pixel y = highest voltage
    endpoint = ys[-1]     # the dipped final point
    # the smoothed peak must approach the real peak, well past the endpoint it
    # used to be clamped to (old bug: sm.min() stuck at ~endpoint)
    assert sm.min() < endpoint - 5.0
    assert sm.min() <= true_peak + 3 * _metrics().line_width_px


# --- confidence weighting ---


def test_low_confidence_span_smoothed_more_than_unweighted():
    from scipy.interpolate import UnivariateSpline

    xs = np.arange(0, 160)
    fn = lambda x: 100.0 + 0.5 * x
    rng = np.random.default_rng(7)
    ys = np.array([fn(x) for x in xs]) + rng.normal(0, 1.0, size=len(xs))
    # inject a noisy, low-confidence (high-cost) bump in the middle span
    ys[70:90] += 8.0
    info = [PointInfo(True, 2, 0.0) for _ in xs]
    for i in range(70, 90):
        info[i] = PointInfo(True, 1, 20.0)  # merged + high cost => low weight
    t = Track("t", "s", "increasing", pixel_sign=+1,
              points_raw=[(int(x), float(y)) for x, y in zip(xs, ys)], info_raw=info)
    m = _metrics()
    smooth_track(t, m, active_track_count=2)
    weighted = np.array([p[1] for p in t.points_smooth])

    # same s, but unweighted spline
    from voltage_tracer.smooth import _local_residuals, _strictly_increasing
    xf = _strictly_increasing(xs.astype(float))
    sigma = max(np.median(_local_residuals(xf, ys, 4)),
                m.d.smooth_sigma_floor_lw * m.line_width_px)
    s = len(xs) * (m.d.smooth_k * sigma) ** 2
    unweighted = UnivariateSpline(xf, ys, s=s)(xf)

    truth = np.array([fn(x) for x in xs])
    span = slice(72, 88)
    assert np.mean((weighted[span] - truth[span]) ** 2) < \
        np.mean((unweighted[span] - truth[span]) ** 2)


def test_weights_deterministic_and_bounded():
    xs = np.arange(0, 60)
    t, _ = _noisy_track(lambda x: 40.0 + x, xs, sign=+1, seed=8)
    m = _metrics()
    w1 = compute_point_weights(t, m, 2)
    w2 = compute_point_weights(t, m, 2)
    assert np.array_equal(w1, w2)
    assert w1.min() >= DEFAULTS.w_floor - 1e-9 and w1.max() <= 1.0 + 1e-9


def test_high_cost_point_gets_lower_weight():
    info = [PointInfo(True, 2, 0.0), PointInfo(True, 2, 20.0), PointInfo(True, 2, 0.0)]
    t = Track("t", "s", "increasing", pixel_sign=+1,
              points_raw=[(0, 10.0), (1, 11.0), (2, 12.0)], info_raw=info)
    w = compute_point_weights(t, _metrics(), 2)
    assert w[1] < w[0] and w[1] < w[2]


def test_merged_column_gets_lower_weight():
    info = [PointInfo(True, 2, 0.0), PointInfo(True, 1, 0.0), PointInfo(True, 2, 0.0)]
    t = Track("t", "s", "increasing", pixel_sign=+1,
              points_raw=[(0, 10.0), (1, 11.0), (2, 12.0)], info_raw=info)
    w = compute_point_weights(t, _metrics(), 2)
    assert w[1] < w[0]


# --- dual-resolution invariance ---


def test_dual_resolution_smoothing_strength_is_scale_invariant():
    """Same curve at 1x and 2x: the smoother's data-space error to truth is
    comparable at both resolutions -- if a pixel literal had leaked into the
    strength, one resolution would be smoothed far more than the other.

    The SAME underlying noisy shape is used at both scales (2x is the 1x
    signal linearly upsampled), so only the resolution differs."""
    fn = lambda cap: 100.0 + 0.3 * cap
    base_cap = np.arange(0, 120)
    rng = np.random.default_rng(11)
    base_noise = rng.normal(0, 1.5, size=len(base_cap))
    errs = {}
    for scale in (1, 2):
        cap = np.linspace(0, 119, 120 * scale)
        truth = fn(cap) * scale
        noise = np.interp(cap, base_cap, base_noise) * scale
        ys = truth + noise
        xs = np.arange(len(cap))
        t = Track("t", "s", "increasing", pixel_sign=+1,
                  points_raw=[(int(x), float(y)) for x, y in zip(xs, ys)],
                  info_raw=[PointInfo(True, 2, 0.0) for _ in xs])
        m = _metrics(line_width=2.0 * scale, w=500 * scale, h=400 * scale)
        smooth_track(t, m, 1)
        sm = np.array([p[1] for p in t.points_smooth])
        # normalized RMS error to truth, dimensionless
        errs[scale] = np.sqrt(np.mean((sm - truth) ** 2)) / (truth.max() - truth.min())
    # strengths match within a modest factor -> no pixel literal leaked
    assert 0.5 < errs[1] / errs[2] < 2.0


def test_smooth_all_groups_by_color():
    data = {
        "plot_bbox_px": [0, 0, 500, 400],
        "x_axis": {"scale": "linear", "ticks_px": [0, 500], "ticks_val": [0, 300]},
        "y_axis": {"scale": "linear", "ticks_px": [400, 0], "ticks_val": [2.5, 4.5]},
        "series": [
            {"id": "r_c", "color_rgb": [255, 0, 0], "label": "c", "direction": "increasing"},
            {"id": "r_d", "color_rgb": [255, 0, 0], "label": "d", "direction": "decreasing"},
        ],
    }
    contract = validate_contract(data)
    xs = np.arange(0, 100)
    t1, _ = _noisy_track(lambda x: 50.0 + x, xs, sign=+1, seed=1)
    t1.series_id = "r_c"
    t2, _ = _noisy_track(lambda x: 350.0 - x, xs, sign=-1, seed=2)
    t2.series_id = "r_d"
    out = smooth_all([t1, t2], _metrics(), contract)
    assert all(t.points_smooth is not None for t in out)
