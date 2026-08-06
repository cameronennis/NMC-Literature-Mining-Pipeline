"""Turn a per-color pixel mask into one or more ordered (x, y) pixel paths.

A single color can contain more than one physical curve (e.g. a charge line
and a discharge line of the same color that cross each other). This module
follows each stroke across columns with a predictive tracker so that
crossings get split back into their separate curves -- continuing straight
through a crossing rather than bouncing onto the other track -- then bridges
any remaining gaps (from a *different* color's line crossing on top and
occluding this one) with smooth interpolation, and finally smooths the whole
path to remove per-column tracking jitter.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, median_filter

_HISTORY = 8  # columns of recent history used to estimate a track's direction


def _recent_slope(xs, ys, k=_HISTORY):
    """Least-squares slope over the last k points, not just the last 2 --
    a single noisy/occluded column shouldn't be able to swing the predicted
    direction enough to cause a track to hop onto a crossing neighbor."""
    n = len(xs)
    if n < 2:
        return 0.0
    k = min(n, k)
    x = np.asarray(xs[-k:], dtype=float)
    y = np.asarray(ys[-k:], dtype=float)
    if x[-1] == x[0]:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return slope


def _cluster_column(rows, gap=4):
    """Group a sorted set of masked row indices in one column into separate
    strokes, splitting wherever consecutive rows are more than `gap` apart."""
    if len(rows) == 0:
        return []
    rows = np.sort(rows)
    clusters = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= gap:
            cur.append(r)
        else:
            clusters.append(np.array(cur))
            cur = [r]
    clusters.append(np.array(cur))
    return clusters


class _Track:
    __slots__ = ("xs", "ys", "missed")

    def __init__(self, x, y):
        self.xs = [x]
        self.ys = [y]
        self.missed = 0

    def predict(self, x):
        slope = _recent_slope(self.xs, self.ys)
        return self.ys[-1] + slope * (x - self.xs[-1])

    def extend(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.missed = 0


def _assign_column(active, x, clusters, tol, max_jump, local_window=6, min_separation=4):
    """Match this column's pixel clusters onto active tracks. When only one
    track's prediction lands in a cluster, that's an unambiguous continuation
    -- though the cluster can still be far wider than one line's true
    thickness (two curves of similar hue touching/blending for a stretch),
    so even then we only trust the pixels near this track's own predicted
    position, not the whole blob's median.

    When two or more tracks' predictions land in the same cluster, two
    different things can be going on, and they need different handling:
    - The curves are merely close (predictions still meaningfully
      separated) -- split the cluster's real pixels by nearest prediction,
      same idea as line-thickness -- this still grounds each track in real
      data every column, so it doesn't drift over a long near-overlap.
    - The curves are truly coincident right at a crossing (predictions
      almost identical) -- there is no pixel position that can tell them
      apart, so splitting by "nearest pixel" there is exactly what causes a
      track to swap identity and bounce off at the crossing instead of
      continuing through. Coast every such track forward on its own
      predicted position for this one column instead, and resume using real
      pixels as soon as the predictions separate again."""
    centroids = [np.median(c) for c in clusters]
    claimed_tracks, claimed_clusters = set(), set()

    for ci, pix_rows in enumerate(clusters):
        cmin, cmax = pix_rows.min(), pix_rows.max()
        matching = [ti for ti, t in enumerate(active)
                    if ti not in claimed_tracks and cmin - tol <= t.predict(x) <= cmax + tol]
        if not matching:
            continue
        claimed_clusters.add(ci)
        if len(matching) == 1:
            ti = matching[0]
            pred = active[ti].predict(x)
            near = pix_rows[np.abs(pix_rows - pred) <= local_window]
            pos = np.median(near) if len(near) else centroids[ci]
            active[ti].extend(x, pos)
            claimed_tracks.add(ti)
        else:
            preds = {ti: active[ti].predict(x) for ti in matching}
            spread = max(preds.values()) - min(preds.values())
            if spread < min_separation:
                for ti in matching:
                    active[ti].extend(x, preds[ti])
                    claimed_tracks.add(ti)
            else:
                nearest = np.array([
                    matching[np.argmin(np.abs(np.array(list(preds.values())) - px))]
                    for px in pix_rows
                ])
                for ti in matching:
                    sub = pix_rows[nearest == ti]
                    if len(sub):
                        active[ti].extend(x, np.median(sub))
                    else:
                        active[ti].extend(x, preds[ti])
                    claimed_tracks.add(ti)

    remaining_tracks = [ti for ti in range(len(active)) if ti not in claimed_tracks]
    remaining_clusters = [ci for ci in range(len(clusters)) if ci not in claimed_clusters]
    for ti in remaining_tracks:
        if not remaining_clusters:
            break
        pred = active[ti].predict(x)
        dists = [abs(pred - centroids[ci]) for ci in remaining_clusters]
        j = int(np.argmin(dists))
        if dists[j] <= max_jump:
            ci = remaining_clusters.pop(j)
            active[ti].extend(x, centroids[ci])
            claimed_tracks.add(ti)
            claimed_clusters.add(ci)

    for ti, t in enumerate(active):
        if ti not in claimed_tracks:
            t.missed += 1

    return [ci for ci in range(len(clusters)) if ci not in claimed_clusters]


def _follow_tracks(mask, cluster_gap=4, tol=3, max_jump=15, max_missed=25):
    h, w = mask.shape
    active, finished = [], []

    for x in range(w):
        rows = np.where(mask[:, x])[0]
        clusters = _cluster_column(rows, gap=cluster_gap)

        if not clusters:
            for t in active:
                t.missed += 1
        else:
            new_idx = _assign_column(active, x, clusters, tol, max_jump)
            centroids = [np.median(clusters[ci]) for ci in new_idx]
            for c in centroids:
                active.append(_Track(x, c))

        still_active = []
        for t in active:
            (finished if t.missed > max_missed else still_active).append(t)
        active = still_active

    finished.extend(active)
    return finished


def _stitch_tracks(tracks, max_gap=60, max_dy=30):
    """Reconnect tracks that were split apart because a different color's
    line fully covered this one for a long stretch of columns."""
    tracks = sorted(tracks, key=lambda t: t.xs[0])
    used = [False] * len(tracks)
    chains = []

    for i, t in enumerate(tracks):
        if used[i]:
            continue
        used[i] = True
        xs, ys = list(t.xs), list(t.ys)
        extended = True
        while extended:
            extended = False
            last_x, last_y = xs[-1], ys[-1]
            slope = _recent_slope(xs, ys)
            best_j, best_cost = None, None
            for j, t2 in enumerate(tracks):
                if used[j] or t2.xs[0] <= last_x:
                    continue
                gap = t2.xs[0] - last_x
                if gap > max_gap:
                    continue
                cost = abs((last_y + slope * gap) - t2.ys[0])
                # A real curve doesn't reverse direction right at a stitch
                # seam. Penalize (don't forbid, a genuine kink is possible)
                # candidates whose own *net* trend (endpoint to endpoint,
                # not a local windowed fit -- the few points right after a
                # dense crossing tangle are often transiently noisy and a
                # poor stand-in for the track's real direction) runs
                # opposite to the ending track's, otherwise the nearest-y
                # match alone can bridge two different physical curves that
                # happen to pass close to the same point (e.g. right where
                # several curves converge near x=0). The penalty is applied
                # before the distance gate (not just for ranking) so a
                # same-direction match a bit further away correctly beats an
                # opposite-direction match that merely happens to be nearer.
                this_dir = np.sign(last_y - ys[0]) if len(ys) >= 2 else 0
                cand_dir = np.sign(t2.ys[-1] - t2.ys[0]) if len(t2.ys) >= 2 else 0
                mismatched = this_dir != 0 and cand_dir != 0 and this_dir != cand_dir
                ranked_cost = cost * 3 if mismatched else cost
                if ranked_cost > max_dy:
                    continue
                if best_cost is None or ranked_cost < best_cost:
                    best_cost, best_j = ranked_cost, j
            if best_j is not None:
                xs.extend(tracks[best_j].xs)
                ys.extend(tracks[best_j].ys)
                used[best_j] = True
                extended = True
        chains.append((np.array(xs, dtype=float), np.array(ys, dtype=float)))

    return chains


def _fill_gaps(x_valid, y_valid):
    """Smoothly interpolate across interior gaps only (leading/trailing
    columns with no data stay empty -- the curve just doesn't extend there)."""
    order = np.argsort(x_valid)
    x_valid, y_valid = x_valid[order], y_valid[order]
    x_valid, unique_idx = np.unique(x_valid, return_index=True)
    y_valid = y_valid[unique_idx]
    if len(x_valid) < 2:
        return x_valid, y_valid
    interp = PchipInterpolator(x_valid, y_valid, extrapolate=False)
    x_full = np.arange(x_valid[0], x_valid[-1] + 1)
    y_full = interp(x_full)
    keep = ~np.isnan(y_full)
    return x_full[keep], y_full[keep]


def _smooth(y, median_window=5, gaussian_sigma=1.5):
    """Light denoising pass: a rolling median first to pull back any
    columns that drifted toward a neighboring line's edge during a crossing
    or near-overlap (an outlier relative to the local trend), then a small
    Gaussian blur so the traced curve reads as smooth rather than jagged,
    without erasing genuine sharp features like a discharge cutoff."""
    n = len(y)
    if n < 3:
        return y
    mw = min(median_window, n if n % 2 == 1 else n - 1)
    if mw >= 3:
        y = median_filter(y, size=mw, mode="nearest")
    return gaussian_filter1d(y, sigma=gaussian_sigma, mode="nearest")


def mask_to_paths(mask, cluster_gap=4, tol=3, max_jump=15, max_missed=25,
                   stitch_gap=60, stitch_max_dy=30, min_length=8,
                   max_start_frac=0.15):
    """Split a color mask into one or more curve paths. Returns a list of
    dicts: {"x": x_px array, "y": y_px array, "direction": "charge"/
    "discharge"/None}. `direction` is only set when 2+ paths were found for
    this color (i.e. there's something to tell apart); it's assigned by the
    sign of the path's overall slope, matching the usual shape of a
    charge/discharge curve (voltage rises during charge, falls during
    discharge).

    Every curve in this domain (voltage vs. capacity) starts at ~0 capacity;
    `max_start_frac` drops any stitched chain that starts later than that
    fraction of the mask's width -- a real curve never legitimately starts
    mid-plot, so a chain that does is a tracking artifact (most often a
    short fragment near a steep drop that failed to stitch back onto its
    real track), not a second real curve.
    """
    h, w = mask.shape
    raw_tracks = _follow_tracks(mask, cluster_gap, tol, max_jump, max_missed)
    raw_tracks = [t for t in raw_tracks if len(t.xs) >= min_length]
    if not raw_tracks:
        return []

    chains = _stitch_tracks(raw_tracks, stitch_gap, stitch_max_dy)
    chains = [(x, y) for x, y in chains if len(x) >= min_length]
    chains = [(x, y) for x, y in chains if x.min() <= max_start_frac * w]

    paths = []
    for x_valid, y_valid in chains:
        x_full, y_full = _fill_gaps(x_valid, y_valid)
        if len(x_full) == 0:
            continue
        y_full = _smooth(y_full)
        slope = np.polyfit(x_full, y_full, 1)[0]
        paths.append({"x": x_full, "y": y_full, "slope": slope})

    if len(paths) >= 2:
        for p in paths:
            p["direction"] = "charge" if p["slope"] >= 0 else "discharge"
    else:
        for p in paths:
            p["direction"] = None
    for p in paths:
        del p["slope"]

    return paths
