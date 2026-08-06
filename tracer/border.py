"""Detect the solid black plot-border rectangle in a graph image."""
import cv2
import numpy as np


def _group_runs(indices):
    """Split a sorted array of indices into contiguous runs, return one
    representative index (the mean) per run."""
    if len(indices) == 0:
        return []
    groups = []
    run = [indices[0]]
    for idx in indices[1:]:
        if idx - run[-1] <= 2:
            run.append(idx)
        else:
            groups.append(run)
            run = [idx]
    groups.append(run)
    return [int(round(np.mean(g))) for g in groups]


def detect_plot_border(gray, dark_thresh=80, frac=0.5):
    """Find the pixel bounding box of the black axis rectangle.

    A border line is a row/column where at least `frac` of pixels are
    darker than `dark_thresh`. Returns (left, right, top, bottom) in
    pixel coordinates.
    """
    h, w = gray.shape
    dark = gray < dark_thresh

    col_frac = dark.mean(axis=0)
    row_frac = dark.mean(axis=1)

    cols = _group_runs(np.where(col_frac > frac)[0])
    rows = _group_runs(np.where(row_frac > frac)[0])

    if len(cols) < 2 or len(rows) < 2:
        raise ValueError(
            f"Could not find a rectangular border (found {len(cols)} vertical "
            f"and {len(rows)} horizontal candidate lines). Try adjusting "
            f"dark_thresh/frac."
        )

    left, right = min(cols), max(cols)
    top, bottom = min(rows), max(rows)
    return left, right, top, bottom


def _find_thin_lines(fracs, frac_thresh=0.5, merge_gap=8, max_thickness=5):
    """Like _group_runs, but (a) merges candidates within `merge_gap` of each
    other instead of just adjacent ones, and (b) discards any merged group
    thicker than `max_thickness`. A true 1-3px border line survives; a big
    filled block (e.g. dense title text, a screenshot tooltip/banner pasted
    over the figure) gets merged into one thick group and dropped."""
    idx = np.where(fracs > frac_thresh)[0]
    if len(idx) == 0:
        return []
    groups = []
    run = [idx[0]]
    for i in idx[1:]:
        if i - run[-1] <= merge_gap:
            run.append(i)
        else:
            groups.append(run)
            run = [i]
    groups.append(run)
    return [int(round(np.mean(g))) for g in groups if (g[-1] - g[0] + 1) <= max_thickness]


def detect_vertical_borders(gray, dark_thresh=80, frac=0.5):
    """Find the left/right pixel columns of the shared plot border. Used for
    both single graphs and a column of stacked graphs, which typically share
    one continuous left/right border spanning the full stack."""
    dark = gray < dark_thresh
    cols = _group_runs(np.where(dark.mean(axis=0) > frac)[0])
    if len(cols) < 2:
        raise ValueError(
            f"Could not find left/right border lines (found {len(cols)}). "
            f"Try adjusting dark_thresh/frac."
        )
    return min(cols), max(cols)


def detect_panel_dividers(gray, left, right, dark_thresh=80, frac=0.5,
                           merge_gap=8, max_thickness=5, expected_count=None):
    """Find every horizontal border line between `left` and `right` (the top
    of the first panel, the boundary between each pair of stacked panels, and
    the bottom of the last one). For N stacked panels this should return N+1
    rows.

    If `expected_count` (= number of panels the caller wants) is given and
    exactly one divider short is found, the top-most divider is assumed to be
    obscured (e.g. by an overlaid caption/screenshot artifact) and is
    estimated from the median spacing of the dividers that were found.
    """
    dark = gray[:, left:right + 1] < dark_thresh
    row_frac = dark.mean(axis=1)
    dividers = _find_thin_lines(row_frac, frac, merge_gap, max_thickness)

    if expected_count is not None and len(dividers) == expected_count:
        heights = np.diff(dividers)
        median_h = float(np.median(heights))
        estimated_top = dividers[0] - median_h
        print(
            f"Warning: only found {len(dividers)}/{expected_count + 1} panel "
            f"dividers; assuming the top-most one is obscured and estimating "
            f"it at row {estimated_top:.0f} from the median panel height "
            f"({median_h:.0f}px). Check the debug image for panel 1."
        )
        dividers = [estimated_top] + dividers

    return [int(round(d)) for d in dividers]


def detect_stacked_panels(gray, expected_count=None, dark_thresh=80, frac=0.5):
    """Return a list of (left, right, top, bottom) pixel boxes, one per
    panel, top to bottom. Works for a single graph (returns a 1-item list)
    or a column of N stacked graphs sharing one left/right border."""
    left, right = detect_vertical_borders(gray, dark_thresh, frac)
    dividers = detect_panel_dividers(
        gray, left, right, dark_thresh, frac, expected_count=expected_count
    )
    if len(dividers) < 2:
        raise ValueError(
            f"Could not find panel dividers (found {len(dividers)}); "
            f"try adjusting detection thresholds."
        )
    return [(left, right, dividers[i], dividers[i + 1]) for i in range(len(dividers) - 1)]


def _boxes_close(a, b, tol=6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _box_area(b):
    return max(0, b[1] - b[0]) * max(0, b[3] - b[2])


def _contains(outer, inner, frac=0.9):
    """True if `inner` sits almost entirely inside `outer` (overlap area is
    at least `frac` of inner's own area)."""
    ox0, oy0 = max(outer[0], inner[0]), max(outer[2], inner[2])
    ox1, oy1 = min(outer[1], inner[1]), min(outer[3], inner[3])
    overlap = max(0, ox1 - ox0) * max(0, oy1 - oy0)
    inner_area = _box_area(inner)
    return inner_area > 0 and overlap / inner_area >= frac


def find_all_boxes(gray, dark_thresh=80, min_size=40, margin=25):
    """Find every rectangular axis-box anywhere in an already-localized
    figure image (a compound figure with several lettered panels, some of
    which may themselves be a vertically-stacked group).

    Two-stage: contours give *approximate* candidate regions cheaply (a
    contour's bounding rect can be looser than the true border whenever
    nearby text/ticks get bridged into it by the morphological close), then
    each candidate is refined by re-running the already-tested precise
    line-scan detector (`detect_plot_border`) on a padded crop around it --
    far more robust than trusting contour geometry directly. Returns
    (left, right, top, bottom) boxes in the full image's frame, deduplicated.
    """
    h, w = gray.shape
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_size or bh < min_size:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if not (4 <= len(approx) <= 12):
            continue

        cx0, cy0 = max(0, x - margin), max(0, y - margin)
        cx1, cy1 = min(w, x + bw + margin), min(h, y + bh + margin)
        crop = gray[cy0:cy1, cx0:cx1]
        try:
            left, right, top, bottom = detect_plot_border(crop, dark_thresh=dark_thresh)
        except ValueError:
            continue
        boxes.append((left + cx0, right + cx0, top + cy0, bottom + cy0))

    boxes.sort(key=_box_area, reverse=True)
    deduped = []
    for b in boxes:
        if not any(_boxes_close(b, kept) for kept in deduped):
            deduped.append(b)

    # A refined box that swallowed a whole stacked column (its crop margin
    # bridged across the shared border between adjacent panels) fully
    # contains 2+ of the smaller, individually-detected panels -- drop the
    # container, keep its contents.
    smaller_first = sorted(deduped, key=_box_area)
    containers = set()
    for i, outer in enumerate(smaller_first):
        contained = sum(
            1 for j, inner in enumerate(smaller_first)
            if i != j and _box_area(inner) < _box_area(outer) and _contains(outer, inner)
        )
        if contained >= 2:
            containers.add(outer)
    return [b for b in smaller_first if b not in containers]
