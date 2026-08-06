"""Per-color mask -> 1px-wide skeleton -> per-column candidate y-positions.

The tracker consumes the output of `extract_column_candidates`: for each
pixel column that has any skeleton pixels, a list of candidate y-centers
(one per vertical cluster). Columns with no pixels are simply absent from
the dict -- deliberately. A missing column is NOT curve termination: gaps
happen at intersections (another color drawn on top), dashed line styles,
and anti-aliased crossings where two colors blend into a third. Bridging
those gaps is the tracker's job (via trajectory prediction), not this
module's -- interpolating here would invent data the tracker couldn't
distinguish from real evidence.
"""
import numpy as np
from skimage.morphology import skeletonize as _sk_skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin a boolean color mask (strokes are typically 2-4px wide) down to
    its 1px-wide centerline skeleton."""
    return _sk_skeletonize(mask.astype(bool))


def extract_column_runs(
    skeleton: np.ndarray, merge_gap: int = 2
) -> dict[int, list[tuple[float, float, float]]]:
    """For each pixel column x with any skeleton pixels, cluster that
    column's pixel rows (splitting where consecutive rows are more than
    `merge_gap` apart) and return each cluster's full vertical extent:
    `{x: [(y_min, y_max, y_mean), ...]}`.

    The tracker's main loop only needs the means (see
    `extract_column_candidates`), but the extents matter for terminal
    drops: a discharge curve's near-vertical final drop occupies a tall
    run in just a few columns, which a mean collapses to a useless
    mid-point -- the run's true span is what the drop-extension pass
    consumes."""
    runs: dict[int, list[tuple[float, float, float]]] = {}
    ys_all, xs_all = np.nonzero(skeleton)
    order = np.argsort(xs_all, kind="stable")
    xs_all, ys_all = xs_all[order], ys_all[order]

    start = 0
    n = len(xs_all)
    while start < n:
        x = xs_all[start]
        end = start
        while end < n and xs_all[end] == x:
            end += 1
        rows = np.sort(ys_all[start:end])
        clusters: list[list[int]] = [[rows[0]]]
        for r in rows[1:]:
            if r - clusters[-1][-1] <= merge_gap:
                clusters[-1].append(r)
            else:
                clusters.append([r])
        runs[int(x)] = [
            (float(c[0]), float(c[-1]), float(np.mean(c))) for c in clusters
        ]
        start = end
    return runs


def extract_column_candidates(
    skeleton: np.ndarray, merge_gap: int = 2
) -> dict[int, list[float]]:
    """Mean y per cluster per occupied column: `{x: [y1, y2, ...]}` --
    the tracker's main-loop input. Thin wrapper over
    `extract_column_runs`; `merge_gap` re-merges near-adjacent skeleton
    pixels that thinning can leave in one column, and must stay small so
    two genuinely distinct curves passing near each other are still
    reported as separate candidates."""
    return {
        x: [mean for _, _, mean in cluster_runs]
        for x, cluster_runs in extract_column_runs(skeleton, merge_gap).items()
    }
