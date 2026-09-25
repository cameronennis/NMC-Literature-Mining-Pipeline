"""Multi-trace column-wise tracking -- the core algorithm.

Each physical curve is one persistent Track. Columns are processed left to
right; every column, each active track predicts its next y from a short
trailing linear fit, and the column's candidate y-positions are assigned to
tracks by solving the assignment problem globally (Hungarian algorithm),
NOT by per-track greedy nearest-match -- greedy matching is exactly what
produces identity swaps when several tracks pass close together at once.

Identity is carried by trajectory prediction plus a monotonicity prior
(charge curves only rise, discharge curves only fall), so same-color
crossings resolve automatically: at the crossing the two predictions pass
through each other and each track keeps its own established direction.

Pixel-space note: image y grows downward, and the contract's directions
("increasing"/"decreasing") are in data space. The mapping between the two
is derived from the contract's y-axis tick calibration (`pixel_orientation`)
rather than assumed.
"""
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from voltage_tracer.contract import AxisScale, Contract, Direction, Series
from voltage_tracer.metrics import FigureMetrics

BIG_PENALTY = 1e5


class PointInfo(NamedTuple):
    """Per-point tracking byproducts, retained (not discarded) so a later
    non-destructive pass (smooth.py) can weight each point by confidence.
    All are values the tracker computes anyway while assigning columns.

    observed:   True if the point is a real matched pixel cluster (or a
                deterministic terminal-drop anchor); False if it was filled
                by pure trajectory prediction across a gap.
    n_clusters: number of candidate y-clusters in this point's column --
                fewer than the active-track count signals a merge (ambiguous).
    cost:       Hungarian prediction-to-candidate distance; large means the
                tracker had to reach for this point (shaky, e.g. at a crossing).
    """
    observed: bool = True
    n_clusters: int = 1
    cost: float = 0.0


@dataclass
class Track:
    id: str
    series_id: str | None            # None while pending (unconfirmed)
    direction: str | None            # data-space "increasing"/"decreasing"
    points_raw: list[tuple[int, float]] = field(default_factory=list)
    info_raw: list[PointInfo] = field(default_factory=list)
    points_smooth: list[tuple[int, float]] | None = None
    # calibrated (capacity, voltage) data, populated by calibrate.py -- pixel
    # traces above stay untouched. data_resampled is the smoothed curve on a
    # uniform capacity grid (for cross-paper averaging).
    data_raw: list[tuple[float, float]] | None = None
    data_smooth: list[tuple[float, float]] | None = None
    data_resampled: list[tuple[float, float]] | None = None
    active: bool = True
    missed: int = 0
    coast_run: int = 0              # consecutive columns coasted through occlusion
    pixel_sign: int = 0              # expected sign of dy_px/dx (0 = unconstrained)
    y_extremum: float = float("nan")  # most-advanced y in the track's direction

    def __post_init__(self) -> None:
        # keep info_raw aligned with points_raw for tracks built directly
        # from a seed chain (or in unit tests) without per-point info
        if len(self.info_raw) < len(self.points_raw):
            self.info_raw += [PointInfo()] * (len(self.points_raw) - len(self.info_raw))

    @property
    def points(self) -> list[tuple[int, float]]:
        """Read alias for the raw trace (back-compat; mutations to the raw
        list should go through advance/_pop_* to keep info_raw in sync)."""
        return self.points_raw

    @property
    def last_x(self) -> int:
        return self.points_raw[-1][0]

    @property
    def last_y(self) -> float:
        return self.points_raw[-1][1]

    def advance(self, x: int, y: float, observed: bool = True,
                n_clusters: int = 1, cost: float = 0.0) -> None:
        """Append a point (and its tracking byproducts) and ratchet the
        directional extremum."""
        self.points_raw.append((x, y))
        self.info_raw.append(PointInfo(observed, n_clusters, cost))
        self.missed = 0
        if self.pixel_sign != 0:
            if np.isnan(self.y_extremum) or (y - self.y_extremum) * self.pixel_sign > 0:
                self.y_extremum = y

    def _pop_tail(self) -> None:
        self.points_raw.pop()
        self.info_raw.pop()

    def _pop_head(self) -> None:
        self.points_raw.pop(0)
        self.info_raw.pop(0)


def pixel_orientation(contract: Contract) -> int:
    """Sign of d(pixel_y)/d(value) for the y-axis: -1 on a standard plot
    (larger voltage = smaller pixel row), +1 if the axis runs the other
    way. Derived from the tick calibration, not assumed."""
    vals = np.asarray(contract.y_axis.ticks_val, dtype=float)
    if contract.y_axis.scale is AxisScale.log:
        vals = np.log10(vals)
    slope = np.polyfit(vals, np.asarray(contract.y_axis.ticks_px, float), 1)[0]
    return 1 if slope > 0 else -1


def _data_sign(direction: Direction | str) -> int:
    return 1 if str(direction).endswith("increasing") else -1


def fit_local_slope(track: Track, window: int = 8) -> float:
    """Least-squares slope dy/dx over the trailing `window` points. Uses the
    points' real x values, so gaps widen the baseline instead of being
    ignored. Returns 0.0 with fewer than 2 points."""
    pts = track.points[-window:]
    if len(pts) < 2:
        return 0.0
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if xs[-1] == xs[0]:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def predict_next_y(track: Track, x_next: int, window: int = 8) -> float:
    """Extrapolate the track's trailing linear fit to column x_next."""
    return track.last_y + fit_local_slope(track, window) * (x_next - track.last_x)


def match_cost(
    predicted_y: float,
    candidate_y: float,
    pixel_sign: int,
    extremum_y: float,
    reversal_tol_px: float,
    penalty: float = BIG_PENALTY,
) -> float:
    """Base cost = |predicted - candidate|. Adds `penalty` when accepting
    the candidate would put the track more than `reversal_tol_px` beyond
    its own directional extremum in the WRONG direction.

    Measuring against the running extremum (not the previous point) makes
    the tolerance cumulative: a slow, sustained drift against the track's
    required direction -- e.g. a mislabeled track riding another curve at
    1-2px per column, which a per-step check never catches -- accumulates
    to the cap within a few columns and gets rejected, while genuine local
    jitter (which oscillates around the extremum rather than marching away
    from it) stays untouched."""
    cost = abs(predicted_y - candidate_y)
    if pixel_sign != 0 and not np.isnan(extremum_y):
        reversal = -(candidate_y - extremum_y) * pixel_sign
        if reversal > reversal_tol_px:
            cost += penalty
    return cost


def _pred_is_occluded(
    track: Track,
    x: int,
    other_runs: dict[int, list[tuple[float, float, float]]],
    band: float,
    window: int,
) -> bool:
    """True if the track's predicted position at column `x` is covered by
    another colour's run there -- evidence the curve is hidden (drawn over),
    not ended. `other_runs` are (lo, hi, mean) intervals of the union of the
    OTHER colour groups."""
    runs = other_runs.get(x)
    if not runs:
        return False
    y_pred = predict_next_y(track, x, window)
    return any(lo - band <= y_pred <= hi + band for lo, hi, _ in runs)


def _expected_start_order(series: list[Series], orient: int) -> list[Series]:
    """Order series by where their curves are expected to start vertically
    (in pixel rows, top first), for seeding tracks at the x-minimum.

    Domain prior: at capacity ~0, decreasing (discharge) curves sit near
    the top of the voltage window and increasing (charge) curves near the
    bottom. Within discharges, higher C-rate starts lower (bigger IR drop);
    within charges, higher C-rate starts higher. `c_rate_rank` orders these
    when provided; contract order is the fallback. `orient` flips the whole
    list for an inverted y-axis."""
    dec = [s for s in series if s.direction is Direction.decreasing]
    inc = [s for s in series if s.direction is Direction.increasing]
    rank = lambda s, default: s.c_rate_rank if s.c_rate_rank is not None else default
    dec.sort(key=lambda s: rank(s, series.index(s)))
    inc.sort(key=lambda s: -rank(s, series.index(s)))
    ordered = dec + inc  # top -> bottom for orient == -1 (standard axis)
    return ordered if orient == -1 else ordered[::-1]


def _build_chains(
    columns: dict[int, list[float]],
    xs: list[int],
    link_tol_per_col: float,
    max_link_gap: int,
) -> list[list[tuple[int, float]]]:
    """Link candidates across the given columns into per-curve chains by
    column-to-column continuity. Unlike pooled y-clustering, this cannot
    split one steeply-descending curve into multiple "clusters" (the y
    tolerance scales with the column gap, so consecutive points of a steep
    curve stay linked), and it cannot merge two curves that stay a few
    pixels apart in every column."""
    chains: list[list[tuple[int, float]]] = []
    for x in xs:
        for y in columns[x]:
            best = None
            for ch in chains:
                lx, ly = ch[-1]
                dx = x - lx
                if dx == 0 or dx > max_link_gap:
                    continue  # already extended this column, or too far back
                d = abs(y - ly)
                if d <= link_tol_per_col * dx and (best is None or d < best[1]):
                    best = (ch, d)
            if best is not None:
                best[0].append((x, float(y)))
            else:
                chains.append([(x, float(y))])
    return chains


def initialize_tracks(
    columns: dict[int, list[float]],
    contract: Contract,
    metrics: FigureMetrics,
    series_ids: list[str] | None = None,
) -> tuple[list[Track], list[Series], list[int]]:
    """Seed one track per series near the x-minimum. Builds continuity
    chains over the first `metrics.init_columns` occupied columns; each
    sustained chain (>= `Defaults.min_chain_points` points) is one
    distinguishable curve at the start, paired with a series via
    `_expected_start_order`. Series left unpaired -- because their curve
    starts later, or is occluded at x-min by the low-capacity tangle --
    stay unseeded, and `track_columns` spawns them from the pending pool
    when their pixels appear. Seeding only what is actually distinguishable
    is deliberate: seeding a series onto another curve's points is how
    identities swap.

    Returns (tracks, unseeded_series, consumed_column_xs)."""
    series = [
        s for s in contract.series
        if series_ids is None or s.id in series_ids
    ]
    orient = pixel_orientation(contract)
    xs = sorted(columns)[: metrics.init_columns]
    chains = [
        ch for ch in _build_chains(
            columns, xs, metrics.link_tol_px, metrics.max_link_gap_cols
        )
        if len(ch) >= metrics.d.min_chain_points
    ]
    # keep the strongest n chains, then pair top-to-bottom with the series
    chains.sort(key=len, reverse=True)
    chains = chains[: len(series)]
    chains.sort(key=lambda ch: float(np.mean([y for _, y in ch])))

    order = list(_expected_start_order(series, orient))
    tracks = []
    for ch in chains:
        if not order:
            break
        # prefer a series whose direction matches the chain's observed
        # slope (decisive when fewer chains than series -- binding purely
        # by position would happily seed a discharge onto a charge curve)
        net = ch[-1][1] - ch[0][1]
        pick = order[0]
        if abs(net) > metrics.line_width_px:
            chain_sign = 1 if net > 0 else -1
            pick = next(
                (s for s in order if _data_sign(s.direction) * orient == chain_sign),
                order[0],
            )
        order.remove(pick)
        sign = _data_sign(pick.direction) * orient
        ys = [y for _, y in ch]
        tracks.append(
            Track(
                id=f"track_{pick.id}",
                series_id=pick.id,
                direction=pick.direction.value,
                points_raw=list(ch),
                pixel_sign=sign,
                y_extremum=max(ys) if sign > 0 else min(ys),
            )
        )
    seeded_ids = {t.series_id for t in tracks}
    unseeded = [s for s in series if s.id not in seeded_ids]
    return tracks, unseeded, xs


def _trim_sparse_tail(
    track: Track, window: int = 15, min_density: float = 0.5
) -> None:
    """Cut a track's trailing points back to where its column support is
    dense. A real curve produces a candidate in nearly every column it
    spans; a track that outlived its curve and wandered onto scattered
    noise (JPEG halos, stray unmasked pixels) matches only sporadically.
    Walking back until the trailing `window`-column span again contains at
    least `min_density * window` matched points removes exactly that
    wandering tail while leaving densely-supported (real) endings alone."""
    while len(track.points_raw) > 2:
        span_start = track.last_x - window
        tail = [p for p in track.points_raw if p[0] > span_start]
        if len(tail) >= min_density * window:
            break
        track._pop_tail()


def _trim_outlier_head(
    track: Track, probe: int = 6, max_residual: float = 12.0
) -> None:
    """Drop a leading point only if it is a genuine POSITIONAL OUTLIER --
    off the line traced by the points that immediately follow it. This is
    the stray-point artifact (a lucky early match or a mis-seed glued to
    the start, rendering as an absurd connecting segment).

    Deliberately NOT sparsity-based: a correct seed head followed by an
    occlusion gap (e.g. a discharge visible for a few columns, hidden while
    another curve crosses over it, then reappearing) is sparse but on-trend,
    and must be kept -- it is the true start of the curve. Only a point
    that both the following trend disowns is removed."""
    while len(track.points_raw) > probe + 1:
        xs = np.array([p[0] for p in track.points_raw[1 : probe + 1]], dtype=float)
        ys = np.array([p[1] for p in track.points_raw[1 : probe + 1]], dtype=float)
        if xs[-1] == xs[0]:
            break
        slope, intercept = np.polyfit(xs, ys, 1)
        hx, hy = track.points_raw[0]
        if abs((slope * hx + intercept) - hy) > max_residual:
            track._pop_head()
        else:
            break


def _trim_kink(track: Track, metrics: FigureMetrics) -> None:
    """Truncate a raw trace where a steep, near-vertical run reverses abruptly
    (a very large second derivative). That kink is the online tracker jogging
    off a terminal drop onto a horizontal annotation or a wrong branch; cutting
    back to the last steep point lets `extend_terminal_drops` follow the true
    slope down. A real curve turns gradually and a real drop bottom only
    decelerates (never reverses), so neither is touched. Runs before the
    terminal-drop pass."""
    d = metrics.d
    pts = track.points_raw
    n = len(pts)
    win = max(2, metrics.slope_window_cols // 2)
    if track.pixel_sign == 0 or n < win + 2:
        return
    s = track.pixel_sign
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    def descent(i0: int, i1: int) -> float:
        """Signed slope from point i0 to i1, positive in the declared dir."""
        return s * (ys[i1] - ys[i0]) / (xs[i1] - xs[i0]) if xs[i1] != xs[i0] else 0.0

    for i in range(win, n - 1):
        # near-vertical descent arriving at i (measured over the trailing window)
        if descent(i - win, i) < d.kink_steep_slope:
            continue
        # ...immediately followed by a reversal against the direction
        ahead = [descent(i, j) for j in range(i + 1, min(n, i + 1 + win))]
        if ahead and min(ahead) <= d.kink_collapse_slope:
            del track.points_raw[i + 1:]
            del track.info_raw[i + 1:]
            ys2 = [y for _, y in track.points_raw]
            if ys2:
                track.y_extremum = max(ys2) if s > 0 else min(ys2)
            return


def _trim_annotation_flats(track: Track, metrics: FigureMetrics) -> None:
    """Excise a dead-horizontal run the track latched onto (a figure annotation
    -- an underline, a retention arrow, a legend rule -- not a real plateau).

    Signature (all pre-smoothing, on the raw pixel trace): the curve CUSPS into
    a run whose pixel-y stays within `flat_band` for at least `flat_min_cols`
    (>= ~5% of plot width), having been clearly OFF that level just before. A
    real voltage plateau fails this -- across such a span it drifts more than a
    stroke width and is approached gradually, not via a cusp. The flat run is
    removed; points after it are kept only while they continue the pre-flat
    trajectory, so a curve that genuinely resumes past the line survives while a
    spurious tail (the common case) is dropped -- then `extend_terminal_drops`
    can descend the real terminal drop from the restored plateau end."""
    pts = track.points_raw
    n = len(pts)
    if n < 4:
        return
    lw = max(1.0, metrics.line_width_px)
    band = metrics.d.flat_band_lw * lw
    cusp_band = metrics.d.flat_cusp_band_lw * lw
    min_span = metrics.flat_min_cols

    a = b = None
    i = 0
    while i < n - 1:
        ymin = ymax = pts[i][1]
        j = i
        while j + 1 < n:
            ny = pts[j + 1][1]
            lo, hi = min(ymin, ny), max(ymax, ny)
            if hi - lo > band:
                break
            ymin, ymax, j = lo, hi, j + 1
        if j > i and (pts[j][0] - pts[i][0]) >= min_span:
            flat_y = 0.5 * (ymin + ymax)
            x_lo = pts[i][0] - min_span
            k = i - 1
            while k >= 0 and pts[k][0] >= x_lo:  # cusp: off the flat level recently
                if abs(pts[k][1] - flat_y) > cusp_band:
                    a, b = i, j
                    break
                k -= 1
            if a is not None:
                break
        i = j + 1
    if a is None:
        return

    keep = list(range(a))  # everything before the flat run
    pre = pts[max(0, a - 8):a]
    xs = [p[0] for p in pre]
    if len(pre) >= 2 and xs[-1] != xs[0]:
        slope, intercept = np.polyfit(xs, [p[1] for p in pre], 1)
        for k in range(b + 1, n):  # keep a genuine continuation past the line
            if abs(pts[k][1] - (slope * pts[k][0] + intercept)) <= metrics.max_jump_px:
                keep.append(k)
            else:
                break
    if len(keep) == n:
        return
    track.points_raw = [pts[k] for k in keep]
    track.info_raw = [track.info_raw[k] for k in keep]
    if track.pixel_sign != 0 and track.points_raw:
        ys = [y for _, y in track.points_raw]
        track.y_extremum = max(ys) if track.pixel_sign > 0 else min(ys)


def extend_terminal_drops(
    tracks: list[Track],
    runs: dict[int, list[tuple[float, float, float]]],
    metrics: FigureMetrics,
) -> None:
    """Extend each track through its near-vertical terminal segment.

    A discharge curve's final drop (and a charge curve's final rise to the
    cutoff voltage) spans hundreds of rows in only 2-4 columns. Per-column
    tracking cannot descend it -- each column collapses to cluster means
    far beyond any sane matching gate -- so this dedicated pass consults
    the full vertical extents from `extract_column_runs` instead: starting
    at the track's end, it chains runs ever further in the track's own
    direction. The column window SLIDES with the last-consumed run (a
    terminal fall is rarely perfectly vertical -- a 20C discharge can fall
    ~18px/column across many columns, far past any fixed window at the
    track's end), and `ext_run_gap_px` of vertical occlusion is tolerated
    between fragments, because other curves cross ON TOP of terminal drops
    routinely. Runaway is prevented by requiring every chained run to
    strictly advance the frontier while staying within `ext_col_offset`
    columns of the previous one -- a neighboring curve's plateau doesn't
    advance, and the next C-rate's drop is many columns away. Only
    committed when the total extension reaches `ext_min_extension_px`, so
    noisy near-end runs don't produce micro-tails.

    All windows/tolerances come from `metrics` (no baked pixel literals).

    Each consumed run contributes one ANCHOR at (its column, its far
    edge); the committed extension is then densified by interpolating x
    AND y together between consecutive anchors at `ext_step_px` spacing.
    This keeps the drop a smooth slanted line even when the runs land on
    sparse columns (a low-res or heavily-occluded drop) -- without it, all
    of a run's vertical progress piles up at that run's single column and
    the trace stair-steps."""
    max_col_offset = metrics.ext_col_offset
    max_run_gap = metrics.ext_run_gap_px
    min_extension = metrics.ext_min_extension_px
    step = metrics.ext_step_px
    min_advance = metrics.ext_min_advance_px
    back_overlap = metrics.ext_back_overlap_px
    for t in tracks:
        s = t.pixel_sign
        if s == 0 or not t.points_raw:
            continue
        x_end, y_end = t.points_raw[-1]
        frontier = float(y_end)
        cur_x = int(x_end)
        anchors: list[tuple[float, float]] = [(float(x_end), float(y_end))]
        progressed = True
        while progressed:
            progressed = False
            best = None  # (advance, x, far_edge)
            for x in range(cur_x - 1, cur_x + max_col_offset + 1):
                for mn, mx, _ in runs.get(x, []):
                    near = mn if s > 0 else mx
                    far = mx if s > 0 else mn
                    advance = (far - frontier) * s
                    gap = (near - frontier) * s
                    if advance > min_advance and -back_overlap <= gap <= max_run_gap:
                        if best is None or advance > best[0]:
                            best = (advance, x, far)
            if best is not None:
                _, x, far = best
                anchors.append((float(x), float(far)))
                frontier = far
                cur_x = max(cur_x, x)
                progressed = True
        if (frontier - y_end) * s < min_extension:
            continue
        for (ax0, ay0), (ax1, ay1) in zip(anchors, anchors[1:]):
            n = max(1, int(abs(ay1 - ay0) / step))
            for k in range(1, n + 1):
                f = k / n
                # terminal-drop points are deterministic anchors from real
                # mask runs -> observed/high-confidence, so smoothing honors
                # the sharp corner without any region special-case
                t.advance(int(round(ax0 + f * (ax1 - ax0))), ay0 + f * (ay1 - ay0),
                          observed=True, n_clusters=1, cost=0.0)


def _gap_is_occluded(
    x_lo: int, y_lo: float, x_hi: int, y_hi: float,
    other_runs: dict[int, list[tuple[float, float, float]]],
    band: float, min_frac: float,
) -> bool:
    """True if a straight line from (x_lo, y_lo) to (x_hi, y_hi) is covered by
    another series' colour over at least `min_frac` of its columns.

    This is the proof that the gap was an OCCLUSION (the curve ran under another
    colour) rather than a real end: at each gap column we look for an other-
    colour run whose vertical extent comes within `band` of the interpolated
    bridge path. An empty gap (no other colour) fails and the bridge is refused."""
    span = x_hi - x_lo
    if span <= 1:
        return False
    covered = 0
    total = 0
    for x in range(x_lo + 1, x_hi):
        total += 1
        yi = y_lo + (y_hi - y_lo) * (x - x_lo) / span
        for mn, mx, _ in other_runs.get(x, []):
            if mn - band <= yi <= mx + band:
                covered += 1
                break
    return total > 0 and covered / total >= min_frac


def bridge_occluded_segments(
    tracks: list[Track],
    columns: dict[int, list[float]],
    other_runs: dict[int, list[tuple[float, float, float]]],
    contract: Contract,
    metrics: FigureMetrics,
) -> None:
    """Stitch a curve back together where a long overlap hid it.

    The online tracker abandons a curve that another colour covers for more
    columns than its coast tolerance, and its visible continuation reappears
    too far away for the pending pool to reclaim. This post-pass repairs that:
    for each confirmed track it looks for a downstream chain of the SAME colour
    (these `columns` are one colour group) heading in the SAME direction, and
    bridges it on when the intervening gap is filled by another colour.

    Safety rests on a structural fact: a colour belongs to exactly one charge
    and one discharge, so within a group every decreasing chain is part of the
    single discharge and every increasing chain part of the single charge --
    a same-direction downstream chain can only be this track's own curve. On
    top of that: the continuation must not reverse the track's direction, the
    gap must be occluded by another colour (not merely empty), the join must be
    gentle (a steep one would be a different feature, e.g. a terminal drop left
    for `extend_terminal_drops`), and the chain must be substantial. Bridge
    points across the gap are interpolated and marked `observed=False`, so the
    smoother down-weights them and validation can flag the reconstructed span.
    Purely additive: a track is only ever extended, never shortened."""
    if not columns:
        return
    xs_all = sorted(columns)
    chains = [
        ch for ch in _build_chains(
            columns, xs_all, metrics.link_tol_px, metrics.max_link_gap_cols
        )
        if len(ch) >= metrics.d.min_chain_points
        and (ch[-1][0] - ch[0][0]) >= metrics.bridge_min_chain_cols
    ]
    band = metrics.max_jump_px
    step = max(1.0, metrics.line_width_px)
    consumed: set[int] = set()

    for t in tracks:
        if t.pixel_sign == 0 or not t.points_raw:
            continue
        progressed = True
        while progressed:
            progressed = False
            best = None  # (x_start, chain)
            for ch in chains:
                if id(ch) in consumed:
                    continue
                cx0, cy0 = ch[0]
                net = ch[-1][1] - ch[0][1]
                if abs(net) < metrics.line_width_px:
                    continue  # too flat to have a definite direction
                if (1 if net > 0 else -1) != t.pixel_sign:
                    continue  # opposite direction -> the other curve in this group
                # must start beyond the track's end, past a real gap
                if cx0 - t.last_x < metrics.bridge_min_gap_cols:
                    continue
                # must continue the track's direction, not reverse it
                if (cy0 - t.last_y) * t.pixel_sign < -metrics.reversal_tol_px:
                    continue
                # gentle join only (a steep one is a drop / different feature)
                slope = abs(cy0 - t.last_y) / max(1, cx0 - t.last_x)
                if slope > metrics.d.bridge_max_slope:
                    continue
                # the gap must be occluded by another colour
                if not _gap_is_occluded(
                    t.last_x, t.last_y, cx0, cy0, other_runs, band,
                    metrics.d.bridge_overlap_frac,
                ):
                    continue
                if best is None or cx0 < best[0]:
                    best = (cx0, ch)
            if best is None:
                continue
            _, ch = best
            gx0, gy0 = t.last_x, t.last_y
            gx1, gy1 = ch[0]
            n = max(1, int((gx1 - gx0) / step))
            for k in range(1, n):  # interpolated gap points (predicted, unseen)
                f = k / n
                t.advance(int(round(gx0 + f * (gx1 - gx0))), gy0 + f * (gy1 - gy0),
                          observed=False, n_clusters=1)
            for cx, cy in ch:  # the recovered visible segment
                t.advance(int(cx), float(cy), observed=True, n_clusters=1)
            consumed.add(id(ch))
            progressed = True


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Best non-decreasing fit to `y` (pool-adjacent-violators, equal weights).
    O(n): each value is pushed as its own block, and adjacent blocks that
    violate monotonicity are merged into their weighted mean."""
    blocks: list[list[float]] = []  # [mean, count]
    for v in y:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            m2, c2 = blocks.pop()
            m1, c1 = blocks.pop()
            blocks.append([(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2])
    out = np.empty(len(y))
    i = 0
    for mean, count in blocks:
        out[i:i + int(count)] = mean
        i += int(count)
    return out


def _directional_scatter(track: Track) -> float:
    """RMS deviation of the raw trace from the best MONOTONE curve in the
    track's declared pixel direction, as a fraction of the trace's vertical
    span. ~0 for any monotone trace regardless of curvature; large only when
    the trace wanders AGAINST its declared direction -- a scattered overlap
    smear rather than a real voltage curve. Returns 0.0 (keep) when there are
    too few points or no vertical extent to judge."""
    ys = np.array([y for _, y in track.points_raw], dtype=float)
    if len(ys) < 4:
        return 0.0
    span = float(ys.max() - ys.min())
    if span <= 0.0:
        return 0.0
    sign = track.pixel_sign or 1
    fit = _isotonic_increasing(sign * ys) * sign  # monotone in declared dir
    resid = float(np.sqrt(np.mean((ys - fit) ** 2)))
    return resid / span


def _adopt_series(track: Track, series: Series, orient: int) -> None:
    """Bind a (pending) track to a series: set its identity, declared
    direction, pixel sign, and directional extremum from its current points."""
    track.series_id = series.id
    track.direction = series.direction.value
    track.pixel_sign = _data_sign(series.direction) * orient
    track.id = f"track_{series.id}"
    ys = [y for _, y in track.points]
    track.y_extremum = max(ys) if track.pixel_sign > 0 else min(ys)


def _extend_head_backward(
    track: Track, columns: dict[int, list[float]], metrics: FigureMetrics,
) -> None:
    """Grow a track's head leftward over columns a rejected incoherent track
    had consumed, re-attaching the branch that continues the track's own trend.
    Follows an already-identified branch, so a monotone-side filter (going
    backward, the pixel row must move opposite the forward direction) keeps it
    off a curve lying the other way, and it tolerates a looser per-column jump
    than forward tracking. Stops after `pending_gap_cols` columns with no
    on-direction candidate -- so a genuinely mid-plot-starting curve is not
    pulled into noise. `columns` holds every candidate (tracking never removed
    them). Points added are real observations."""
    if len(track.points_raw) < 2 or track.pixel_sign == 0:
        return
    window = metrics.slope_window_cols
    lead_jump = metrics.d.lead_extend_jump_mult * metrics.max_jump_px
    max_missed = metrics.pending_gap_cols
    tol = metrics.reversal_tol_px
    xs_left = sorted((x for x in columns if x < track.points_raw[0][0]),
                     reverse=True)
    missed = 0
    for x in xs_left:
        head = track.points_raw[:window]
        hx = np.array([p[0] for p in head], dtype=float)
        hy = np.array([p[1] for p in head], dtype=float)
        slope = float(np.polyfit(hx, hy, 1)[0]) if hx[-1] != hx[0] else 0.0
        x0, y0 = track.points_raw[0]
        pred = y0 + slope * (x - x0)
        best = None
        for y in columns[x]:
            # backward monotonicity: the branch continues opposite the forward
            # direction, so exclude candidates on the wrong side (e.g. the
            # discharge sitting above a charge) -- this, not the jump gate, is
            # what keeps the lead-in on the right curve.
            if track.pixel_sign < 0 and y < y0 - tol:
                continue
            if track.pixel_sign > 0 and y > y0 + tol:
                continue
            d = abs(y - pred)
            if d <= lead_jump and (best is None or d < best[1]):
                best = (y, d)
        if best is None:
            missed += 1
            if missed > max_missed:
                break
            continue
        missed = 0
        track.points_raw.insert(0, (int(x), float(best[0])))
        if track.info_raw is not None:
            track.info_raw.insert(0, PointInfo(True, len(columns[x]), 0.0))
    ys = [y for _, y in track.points_raw]
    track.y_extremum = max(ys) if track.pixel_sign > 0 else min(ys)


def _trim_coasted_tail(track: Track) -> None:
    """Drop trailing UNOBSERVED points (coasted through occlusion). A coast is
    only justified if the curve reappeared after it -- points still hold the
    curve. A track that coasted PAST its true end into a nearby other-colour
    region leaves speculative points with no real pixel after them; remove them
    so the track ends at real data (and is not then judged incoherent). Internal
    coasts, followed by observed points, are untouched."""
    popped = False
    while track.points_raw and not track.info_raw[-1].observed:
        track._pop_tail()
        popped = True
    if popped and track.points_raw and track.pixel_sign != 0:
        ys = [y for _, y in track.points_raw]
        track.y_extremum = max(ys) if track.pixel_sign > 0 else min(ys)


def _reclaim_from_pending(
    pending: list[Track], freed: list[Series], orient: int,
    columns: dict[int, list[float]], metrics: FigureMetrics,
) -> list[Track]:
    """Try to fill each series freed by the coherence gate with a leftover
    pending run -- "try for another curve". A candidate must be long enough,
    move in the series' declared direction, and itself pass the coherence gate;
    otherwise the series is simply left without a curve (better a gap than a
    scattered one). A reclaimed run that starts mid-plot is then extended back
    over the columns the rejected track had consumed."""
    reclaimed: list[Track] = []
    avail = list(freed)
    for p in sorted(pending, key=lambda p: len(p.points), reverse=True):
        if not avail or len(p.points) < metrics.confirm_run_cols:
            continue
        dx = p.last_x - p.points[0][0]
        if dx < metrics.coherence_min_span_cols:
            continue  # too short to stand in as a real curve
        net_sign = int(np.sign(p.last_y - p.points[0][1])) if dx else 0
        pick = next(
            (s for s in avail if _data_sign(s.direction) * orient == net_sign),
            None,
        )
        if pick is None:
            continue
        _adopt_series(p, pick, orient)
        if _directional_scatter(p) > metrics.d.coherence_max_scatter:
            p.series_id = None  # the alternative is no better -- leave it out
            continue
        _extend_head_backward(p, columns, metrics)
        avail.remove(pick)
        reclaimed.append(p)
    return reclaimed


def track_columns(
    columns: dict[int, list[float]],
    contract: Contract,
    metrics: FigureMetrics,
    series_ids: list[str] | None = None,
    other_runs: dict[int, list[tuple[float, float, float]]] | None = None,
) -> list[Track]:
    """Track every series (optionally restricted to `series_ids` -- the
    caller groups series by shared color and passes one group per call)
    through the column candidates. Returns confirmed tracks only.

    Every threshold is taken from `metrics` (derived from the figure), not
    a baked literal: the slope-fit window, per-column match gate, cumulative
    reversal tolerance, deactivation and confirmation column counts, and
    seeding parameters all scale with the figure's measured dimensions and
    stroke width.

    Per column: Hungarian assignment between active-track predictions and
    candidates (costs from `match_cost`, distance-gated by `max_jump_px`);
    matched tracks extend, unmatched tracks miss (deactivated past
    `deactivate_cols` -- a curve legitimately ending), unmatched candidates
    feed a pending pool that can spawn a track for a series whose curve
    starts mid-plot, once it survives `confirm_run_cols` columns."""
    if not columns:
        return []
    orient = pixel_orientation(contract)
    window = metrics.slope_window_cols
    max_jump = metrics.max_jump_px
    reversal_tol_px = metrics.reversal_tol_px
    max_missed = metrics.deactivate_cols
    min_confirm_run = metrics.confirm_run_cols
    pending_gap = metrics.pending_gap_cols
    max_coast = metrics.coast_max_cols

    tracks, unseeded, init_xs = initialize_tracks(columns, contract, metrics, series_ids)
    series_by_id = {
        s.id: s for s in contract.series
        if series_ids is None or s.id in series_ids
    }
    pending: list[Track] = []
    pending_seq = 0

    for x in sorted(columns):
        if x in init_xs:
            continue
        cands = list(columns[x])
        active = [t for t in tracks if t.active]

        matched_cand_idx: set[int] = set()
        if active and cands:
            cost = np.zeros((len(active), len(cands)))
            for r, t in enumerate(active):
                pred = predict_next_y(t, x, window)
                for c, y in enumerate(cands):
                    cost[r, c] = match_cost(
                        pred, y, t.pixel_sign, t.y_extremum, reversal_tol_px
                    )
                    if abs(pred - y) > max_jump:
                        cost[r, c] += BIG_PENALTY
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < BIG_PENALTY / 2:
                    # retain the assignment cost + column cluster count as
                    # per-point confidence byproducts for smooth.py
                    active[r].advance(x, float(cands[c]), observed=True,
                                      n_clusters=len(cands), cost=float(cost[r, c]))
                    active[r].coast_run = 0  # re-acquired a real pixel
                    matched_cand_idx.add(c)

        for t in active:
            if not t.points_raw or t.last_x != x:
                # Occlusion-aware coast: if another colour covers where this
                # track would be, it is hidden (drawn over), not ended -- follow
                # its prediction to stay alive and re-acquire when it reappears.
                # Bounded by `max_coast` so a curve that truly ended (no colour
                # over it) still deactivates. Coasted points are unobserved.
                if (other_runs is not None and t.points_raw
                        and t.coast_run < max_coast
                        and _pred_is_occluded(t, x, other_runs, max_jump, window)):
                    t.advance(x, float(predict_next_y(t, x, window)),
                              observed=False, n_clusters=1)
                    t.coast_run += 1
                    continue
                t.missed += 1
                t.coast_run = 0
                if t.missed > max_missed:
                    t.active = False
                    # A seed that never ESTABLISHED a real trajectory releases
                    # its series back to the unseeded pool, so the real branch
                    # -- once it separates cleanly from the start-of-plot fan --
                    # can claim it from the pending pool instead of being
                    # dropped. "Never established" = it died having covered only
                    # a short x-span (a seed bound onto the merged fan that then
                    # stalled), regardless of how many points that short span
                    # holds. A genuine curve, even a fast-rate one, spans far
                    # more before ending.
                    span = t.last_x - t.points_raw[0][0]
                    if t.series_id is not None and span < metrics.deactivate_cols:
                        s = series_by_id.get(t.series_id)
                        if s is not None and s not in unseeded:
                            unseeded.append(s)
                        t.series_id = None  # orphan the stub; dropped at return

        # ---- pending pool: unclaimed candidates may be a late-starting curve
        leftover = [y for i, y in enumerate(cands) if i not in matched_cand_idx]
        for y in leftover:
            best = None
            for p in pending:
                if x - p.last_x > pending_gap:
                    continue
                d = abs(predict_next_y(p, x, window) - y)
                if d <= max_jump and (best is None or d < best[1]):
                    best = (p, d)
            if best is not None:
                best[0].advance(x, float(y), observed=True, n_clusters=len(cands))
            else:
                pending_seq += 1
                pending.append(
                    Track(id=f"pending_{pending_seq}", series_id=None,
                          direction=None, points_raw=[(x, float(y))])
                )

        still_pending = []
        for p in pending:
            if p.last_x != x:
                p.missed += 1
            if p.missed > pending_gap:
                continue  # short-lived noise (stray unmasked pixels): drop
            if len(p.points) >= min_confirm_run:
                dx = p.last_x - p.points[0][0]
                net_sign = int(np.sign(p.last_y - p.points[0][1])) if dx else 0
                # bind ONLY to an unseeded series whose direction matches the
                # run's observed slope -- never force a wrong-direction series
                pick = next(
                    (s for s in unseeded
                     if _data_sign(s.direction) * orient == net_sign),
                    None,
                )
                if pick is not None:
                    unseeded.remove(pick)
                    _adopt_series(p, pick, orient)
                    tracks.append(p)
                else:
                    # confirmed but no matching series free yet -- keep it
                    # alive; a dying stub may release its series shortly
                    still_pending.append(p)
            else:
                still_pending.append(p)
        pending = still_pending

    # drop orphaned stubs whose series was released to a cleaner branch
    tracks = [t for t in tracks if t.series_id is not None]
    for t in tracks:
        _trim_coasted_tail(t)
        _trim_kink(t, metrics)
        _trim_annotation_flats(t, metrics)
        _trim_sparse_tail(t, window=metrics.trim_window_cols)
        _trim_outlier_head(t, probe=metrics.head_probe_cols,
                           max_residual=metrics.head_residual_px)

    # Coherence gate: a finished track that wanders against its declared
    # direction far more than its own vertical span is a scattered overlap
    # smear, not a real curve. Drop it; free its series and try to reclaim a
    # cleaner run left in the pending pool ("try for another curve"). Done
    # after the trims so a curve isn't judged on a stray head/tail it just shed.
    kept: list[Track] = []
    freed: list[Series] = []
    min_span = metrics.coherence_min_span_cols
    for t in tracks:
        span = t.last_x - t.points_raw[0][0] if t.points_raw else 0
        if (span >= min_span
                and _directional_scatter(t) > metrics.d.coherence_max_scatter):
            s = series_by_id.get(t.series_id)
            if s is not None:
                freed.append(s)
        else:
            kept.append(t)
    if freed:
        kept.extend(_reclaim_from_pending(pending, freed, orient, columns, metrics))
    return kept
