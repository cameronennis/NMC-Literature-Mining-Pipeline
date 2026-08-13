"""Post-hoc sanity checks that flag (never block) suspect traces.

Every check returns structured `Finding` objects and NEVER raises, so a batch
run can aggregate them into a per-figure triage report without halting. These
findings are the >=95%-uncertainty set a human reviews.

Checks run primarily on the RAW pixel trace, so trace smoothing can never
hide a violation; the one exception, `check_smoothing_deviation`, compares
raw against smoothed on purpose -- to surface a place the smoother bridged
something real instead of denoising it.

Ranges are reported in capacity (mAh/g) when the track has been calibrated,
else in pixel-x, so a reviewer can jump straight to the spot.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from voltage_tracer.contract import Contract
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.smooth import _local_residuals, _strictly_increasing
from voltage_tracer.tracker import Track

WARNING = "warning"
VIOLATION = "violation"


@dataclass(frozen=True)
class Finding:
    severity: str                       # WARNING | VIOLATION
    check: str                          # which check produced it
    curve_id: str | None
    description: str
    x_range: tuple[float, float] | None  # capacity (or pixel-x) span, if localized


def _cap_at(track: Track, i: int) -> float:
    """Capacity at raw-point index i (data units if calibrated, else px-x)."""
    if track.data_raw is not None and i < len(track.data_raw):
        return float(track.data_raw[i][0])
    return float(track.points_raw[i][0])


def _reversal_runs(ys: np.ndarray, sign: int):
    """Yield (start, end, max_backtrack) for each maximal run of points that
    sit below the running directional extremum (a reversal against `sign`)."""
    ext = ys[0]
    start = None
    back = 0.0
    for i in range(1, len(ys)):
        progress = (ys[i] - ext) * sign
        if progress >= 0:
            if start is not None:
                yield start, i - 1, back
                start = None
            ext = ys[i]
        else:
            if start is None:
                start, back = i, 0.0
            back = max(back, -progress)
    if start is not None:
        yield start, len(ys) - 1, back


def check_monotonicity(track: Track, metrics: FigureMetrics) -> list[Finding]:
    """Voltage-vs-capacity is monotone per half-cycle. Flag a SUSTAINED
    reversal in the declared pixel direction -- one that backtracks more than
    a stroke width over several points -- while tolerating sub-stroke jitter
    and genuine faint plateaus (which don't backtrack)."""
    if track.pixel_sign == 0 or len(track.points_raw) < metrics.d.mono_min_run + 1:
        return []
    ys = np.array([p[1] for p in track.points_raw], dtype=float)
    tol = metrics.d.mono_tol_lw * metrics.line_width_px
    out: list[Finding] = []
    for a, b, back in _reversal_runs(ys, track.pixel_sign):
        if back > tol and (b - a + 1) >= metrics.d.mono_min_run:
            out.append(Finding(
                VIOLATION, "monotonicity", track.series_id,
                f"sustained {back:.1f}px reversal against declared direction "
                f"over {b - a + 1} points",
                (_cap_at(track, a), _cap_at(track, b)),
            ))
    return out


def check_capacity_ordering(
    tracks: list[Track], contract: Contract, metrics: FigureMetrics
) -> list[Finding]:
    """Higher C-rate generally means lower max capacity. When `c_rate_rank`
    is supplied, warn (never error) if two curves of the same direction have
    max capacities ordered against their ranks by more than a slack."""
    ranks = {s.id: s.c_rate_rank for s in contract.series}
    per_dir: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    all_caps: list[float] = []
    for t in tracks:
        r = ranks.get(t.series_id)
        if r is None or not t.data_raw:
            continue
        max_cap = max(c for c, _ in t.data_raw)
        per_dir[t.direction].append((r, max_cap, t.series_id))
        all_caps.append(max_cap)
    if not all_caps:
        return []
    tol = metrics.d.capacity_order_tol_frac * (max(all_caps) - min(all_caps))
    out: list[Finding] = []
    for direction, items in per_dir.items():
        items.sort(key=lambda it: it[0])  # by C-rate rank ascending
        for (ra, ca, ida), (rb, cb, idb) in zip(items, items[1:]):
            # ra < rb => a is lower C-rate => should have >= capacity
            if ra < rb and ca < cb - tol:
                out.append(Finding(
                    WARNING, "capacity_ordering", idb,
                    f"{idb} (rank {rb}) has higher max capacity "
                    f"({cb:.1f}) than lower-rate {ida} (rank {ra}, {ca:.1f}) "
                    f"-- check C-rate/color assignment",
                    None,
                ))
    return out


def check_terminal_drop_sanity(
    track: Track, metrics: FigureMetrics, contract: Contract
) -> list[Finding]:
    """Flag a trace that runs past the plot box or into a detected text
    region -- the failure mode where a terminal-drop extension overshoots
    into the title/label band or off the axis."""
    out: list[Finding] = []
    h, w = metrics.plot_h_px, metrics.plot_w_px
    for i, (x, y) in enumerate(track.points_raw):
        if not (0 <= y <= h and 0 <= x <= w):
            out.append(Finding(
                VIOLATION, "terminal_drop_sanity", track.series_id,
                f"trace point ({x:.0f},{y:.0f}) lies outside the plot box "
                f"{w}x{h}px -- extension overshoot",
                (_cap_at(track, i), _cap_at(track, i)),
            ))
            break
    for (tx0, ty0, tx1, ty1) in metrics.text_regions:
        inside = [i for i, (x, y) in enumerate(track.points_raw)
                  if tx0 <= x <= tx1 and ty0 <= y <= ty1]
        if inside:
            out.append(Finding(
                VIOLATION, "terminal_drop_sanity", track.series_id,
                f"trace enters detected text region "
                f"({tx0},{ty0})-({tx1},{ty1}) -- likely traced a label",
                (_cap_at(track, inside[0]), _cap_at(track, inside[-1])),
            ))
    return out


def check_smoothing_deviation(track: Track, metrics: FigureMetrics) -> list[Finding]:
    """Surface where the smoothed trace departs from the raw trace by more
    than a measured-noise multiple -- i.e. where the smoother is bridging a
    real feature rather than denoising. Warns; the raw trace stays canonical."""
    if track.points_smooth is None or len(track.points_raw) < metrics.d.deviation_min_run:
        return []
    xs = np.array([p[0] for p in track.points_raw], dtype=float)
    yr = np.array([p[1] for p in track.points_raw], dtype=float)
    ysm = np.array([p[1] for p in track.points_smooth], dtype=float)
    win = max(2, metrics.slope_window_cols // 2)
    sigma = max(float(np.median(_local_residuals(_strictly_increasing(xs), yr, win))),
                metrics.d.smooth_sigma_floor_lw * metrics.line_width_px)
    thresh = metrics.d.smooth_deviation_k * sigma
    over = np.abs(yr - ysm) > thresh
    out: list[Finding] = []
    i = 0
    n = len(over)
    while i < n:
        if over[i]:
            j = i
            while j < n and over[j]:
                j += 1
            if (j - i) >= metrics.d.deviation_min_run:
                out.append(Finding(
                    WARNING, "smoothing_deviation", track.series_id,
                    f"smoothed trace departs from raw by >{thresh:.1f}px over "
                    f"{j - i} points -- smoother may be bridging a real feature",
                    (_cap_at(track, i), _cap_at(track, j - 1)),
                ))
            i = j
        else:
            i += 1
    return out


def validate_all(
    tracks: list[Track], contract: Contract, metrics: FigureMetrics
) -> list[Finding]:
    """Run every check over every track; return all findings (never raises)."""
    findings: list[Finding] = []
    for t in tracks:
        findings += check_monotonicity(t, metrics)
        findings += check_terminal_drop_sanity(t, metrics, contract)
        findings += check_smoothing_deviation(t, metrics)
    findings += check_capacity_ordering(tracks, contract, metrics)
    return findings
