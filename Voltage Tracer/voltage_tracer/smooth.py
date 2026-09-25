"""Non-destructive trace smoothing.

Removes pixel-quantization jitter from each raw trace, producing a separate
`points_smooth` layer WITHOUT touching `points_raw`. The raw column-wise
trace stays the audit ground truth; the smoothed layer is derived and, being
kept alongside the raw, lets any hidden inaccuracy stay visible rather than
be polished away.

Operates in pixel space, before calibration -- pixel jitter is roughly
homoscedastic there, so a noise-driven smoother is best behaved, and
calibration then stays a pure affine map on already-clean points. This is a
decoupled post-pass: it never feeds back into tracking or curve identity.

Strength is self-tuning: the spline's smoothing factor is derived from each
track's own measured scatter (sigma), and each point is weighted by
confidence built from the tracker's retained byproducts -- shaky points
(merged/high-cost columns) are down-weighted, so the spline rides the clean
surrounding trend straight through an intersection tangle without any region
being named or excluded.
"""
import warnings

import numpy as np
from scipy.interpolate import UnivariateSpline

from voltage_tracer.contract import Contract
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.tracker import Track


def _strictly_increasing(xs: np.ndarray) -> np.ndarray:
    """Nudge any duplicate/decreasing x up by a hair -- a near-vertical
    terminal drop can round two anchor points onto one column, and
    UnivariateSpline requires strictly increasing x."""
    out = xs.astype(float).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-6
    return out


def _local_residuals(xs: np.ndarray, ys: np.ndarray, win: int) -> np.ndarray:
    """Per-point residual about a short local linear fit -- the measured
    pixel scatter. Robust (used via median), so the few large residuals at
    a steep terminal drop don't inflate the track's noise estimate."""
    n = len(xs)
    res = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - win), min(n, i + win + 1)
        if b - a < 3:
            continue
        slope, intercept = np.polyfit(xs[a:b], ys[a:b], 1)
        res[i] = abs(ys[i] - (slope * xs[i] + intercept))
    return res


def compute_point_weights(
    track: Track, metrics: FigureMetrics, active_track_count: int
) -> np.ndarray:
    """Per-point confidence weights in [floor, 1] from the tracker's
    retained byproducts. Pure arithmetic; no LLM, no tokens.

    - observed vs bridged: a predicted-across-a-gap point is trusted less.
    - cluster count vs active tracks: a column with fewer y-clusters than
      the tracks sharing this color is a merge -> ambiguous -> lower.
    - match cost: a point the tracker had to reach for (far from its
      prediction, e.g. at a crossing) is shakier, normalized by line width.
    Combined multiplicatively; each factor clamped so no single signal can
    zero a point out."""
    d = metrics.d
    info = track.info_raw
    observed = np.array([1.0 if i.observed else d.w_bridged_floor for i in info])

    ncl = np.array([i.n_clusters for i in info], dtype=float)
    ambiguity = np.clip(ncl / max(1, active_track_count), 0.0, 1.0)
    cluster = d.w_cluster_floor + (1.0 - d.w_cluster_floor) * ambiguity

    cost = np.array([i.cost for i in info], dtype=float)
    denom = max(1e-6, d.w_cost_lw * metrics.line_width_px)
    prediction = np.clip(1.0 - cost / denom, d.w_cost_floor, 1.0)

    return np.clip(observed * cluster * prediction, d.w_floor, 1.0)


def smooth_track(
    track: Track,
    metrics: FigureMetrics,
    active_track_count: int = 1,
) -> Track:
    """Populate `track.points_smooth` from `points_raw` via one mild
    weighted `UnivariateSpline` (smoothing factor from the track's measured
    sigma, weights from `compute_point_weights`), endpoints pinned, result
    made monotone in the track's declared pixel direction. A single uniform
    pass -- no region is excluded; high-confidence terminal-drop anchors
    keep the sharp cutoff corner automatically. `points_raw` is untouched."""
    raw = track.points_raw
    n = len(raw)
    if n < 4:
        track.points_smooth = list(raw)
        return track

    xs = np.array([p[0] for p in raw], dtype=float)
    ys = np.array([p[1] for p in raw], dtype=float)
    xf = _strictly_increasing(xs)

    win = max(2, metrics.slope_window_cols // 2)
    resid = _local_residuals(xf, ys, win)
    sigma = max(float(np.median(resid)),
                metrics.d.smooth_sigma_floor_lw * metrics.line_width_px)

    w = compute_point_weights(track, metrics, active_track_count).copy()
    w[0] = w[-1] = w.max() * metrics.d.smooth_endpoint_w  # pin the endpoints

    s = n * (metrics.d.smooth_k * sigma) ** 2
    with warnings.catch_warnings():
        # a track whose shape can't hit the target `s` within FITPACK's
        # iteration cap still returns a valid (slightly under-smoothed)
        # approximation -- fine here, and not worth spamming a batch log
        warnings.simplefilter("ignore")
        spline = UnivariateSpline(xf, ys, w=w, s=s)
        ysm = spline(xf)

    # monotone-project in the declared pixel direction, then pin the exact
    # raw endpoints. Bound the projection by the raw trace's full extent, not
    # just its two endpoints: when the final raw point is a small dip rather
    # than the curve's extreme (a slight terminal hook), endpoint-only bounds
    # put the clip floor above the true peak and clamp it down to the dip --
    # the "smoothed tail flatlines" bug. The full-extent bound keeps a genuine
    # terminal-drop corner intact (there the endpoint IS the extreme) while
    # letting a real interior peak stand.
    lo, hi = float(ys.min()), float(ys.max())
    if track.pixel_sign > 0:
        ysm = np.maximum.accumulate(ysm)
    elif track.pixel_sign < 0:
        ysm = np.minimum.accumulate(ysm)
    ysm = np.clip(ysm, lo, hi)

    # Pin the endpoints to the raw trace. The start pins exactly; the end pins
    # exactly too UNLESS the raw last point opposes the declared direction (a
    # slight terminal hook / dip), in which case it is clamped to the monotone
    # value. A genuine terminal drop still pins exactly -- there the raw
    # endpoint IS the monotone extreme -- but a hook no longer flattens the
    # tail (the old clip-to-endpoint behavior) nor breaks monotonicity.
    ysm[0] = ys[0]
    if track.pixel_sign > 0:
        ysm[-1] = max(ys[-1], ysm[-2])
    elif track.pixel_sign < 0:
        ysm[-1] = min(ys[-1], ysm[-2])
    else:
        ysm[-1] = ys[-1]

    track.points_smooth = [(int(rx), float(sy)) for rx, sy in zip(xs, ysm)]
    return track


def smooth_all(
    tracks: list[Track], metrics: FigureMetrics, contract: Contract
) -> list[Track]:
    """Smooth every track. `active_track_count` for the cluster-ambiguity
    factor is the number of tracks sharing a curve's color (the set that
    can merge into one column)."""
    color_of = {s.id: s.color_rgb for s in contract.series}
    group_size: dict[tuple[int, int, int], int] = {}
    for t in tracks:
        c = color_of.get(t.series_id)
        if c is not None:
            group_size[c] = group_size.get(c, 0) + 1
    for t in tracks:
        n_group = group_size.get(color_of.get(t.series_id), 1)
        smooth_track(t, metrics, active_track_count=n_group)
    return tracks
