"""Metrics + the no-baked-constants guarantee.

The headline test renders the SAME synthetic figure at 1x and 2x pixel
scale and requires the recovered traces to be identical in normalized
(fraction-of-plot) coordinates. Any surviving pixel literal in the
preprocess/skeletonize/track chain would shift the 2x result and fail this
-- that is the mechanical proof the constants sweep is complete.
"""
import cv2
import numpy as np
import pytest

from voltage_tracer.contract import validate_contract
from voltage_tracer.metrics import compute_metrics, detect_text_regions
from voltage_tracer.defaults import DEFAULTS
from voltage_tracer.preprocess import (
    color_mask,
    crop_to_plot_area,
    mask_gridlines_and_axes,
    mask_regions,
    remove_small_components,
)
from voltage_tracer.skeletonize import extract_column_runs, skeletonize_mask
from voltage_tracer.tracker import extend_terminal_drops, track_columns

_CURVE_RGB = (200, 60, 60)


def _make_figure(scale: int):
    """Two same-color curves (a rising charge, a falling discharge) that
    cross once, drawn at `scale`x resolution with a black border, plus a
    little title text. Returns (image_rgb, contract)."""
    w, h = 400 * scale, 300 * scale
    m = 30 * scale
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    px0, py0, px1, py1 = m, m, w - m, h - m
    pw, ph = px1 - px0, py1 - py0
    thick = max(1, 2 * scale)

    ts = np.linspace(0, 1, 200)
    charge = [(int(px0 + t * pw), int(py1 - (0.15 + 0.7 * t) * ph)) for t in ts]
    discharge = [(int(px0 + t * pw), int(py0 + (0.12 + 0.7 * t) * ph)) for t in ts]
    for pts in (charge, discharge):
        cv2.polylines(img, [np.array(pts, np.int32)], False, _CURVE_RGB, thick, cv2.LINE_AA)
    cv2.rectangle(img, (px0, py0), (px1, py1), (0, 0, 0), thick)
    # a title inside the top of the plot, to exercise text detection
    cv2.putText(img, "CELL A", (px0 + pw // 3, py0 + int(0.06 * ph)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), max(1, scale))

    contract = validate_contract({
        "plot_bbox_px": [px0, py0, px1, py1],
        "x_axis": {"scale": "linear", "ticks_px": [px0, px1], "ticks_val": [0, 100]},
        "y_axis": {"scale": "linear", "ticks_px": [py1, py0], "ticks_val": [2.5, 4.5]},
        "series": [
            {"id": "chg", "color_rgb": list(_CURVE_RGB), "label": "charge",
             "direction": "increasing"},
            {"id": "dis", "color_rgb": list(_CURVE_RGB), "label": "discharge",
             "direction": "decreasing"},
        ],
    })
    return img, contract


def _trace_normalized(scale: int) -> dict[str, np.ndarray]:
    """Full chain on the scale-x figure; return each series' trace resampled
    onto a common normalized-x grid as normalized y (fraction of plot)."""
    img, contract = _make_figure(scale)
    crop = crop_to_plot_area(img, contract.plot_bbox_px)
    metrics = compute_metrics(crop, contract)
    exclude = mask_gridlines_and_axes(crop) | mask_regions(
        crop.shape[:2], metrics.text_regions
    )
    cmask = remove_small_components(
        color_mask(crop, _CURVE_RGB, tolerance=metrics.color_tol) & ~exclude,
        min_area=metrics.speckle_min_area,
    )
    runs = extract_column_runs(skeletonize_mask(cmask), merge_gap=metrics.merge_gap)
    columns = {x: [me for _, _, me in rs] for x, rs in runs.items()}
    tracks = track_columns(columns, contract, metrics, series_ids=["chg", "dis"])
    extend_terminal_drops(tracks, runs, metrics)

    h, w = crop.shape[:2]
    grid = np.linspace(0.15, 0.85, 40)  # sample the shared interior domain
    out = {}
    for t in tracks:
        xs = np.array([p[0] for p in t.points]) / w
        ys = np.array([p[1] for p in t.points]) / h
        order = np.argsort(xs)
        out[t.series_id] = np.interp(grid, xs[order], ys[order])
    return out


def test_metrics_measures_sane_values():
    img, contract = _make_figure(1)
    crop = crop_to_plot_area(img, contract.plot_bbox_px)
    m = compute_metrics(crop, contract)
    assert m.plot_w_px == 340 and m.plot_h_px == 240
    assert 1.0 <= m.line_width_px <= 5.0
    assert m.px_per_xunit == pytest.approx(340 / 100)
    # single color -> separation is infinite, tol falls back to the cap
    assert m.color_tol == DEFAULTS.color_tol_abs_cap


def test_line_width_scales_with_resolution():
    m1 = compute_metrics(*(_prep(1)))
    m2 = compute_metrics(*(_prep(2)))
    assert m2.line_width_px > m1.line_width_px  # thicker strokes at 2x


def _prep(scale):
    img, contract = _make_figure(scale)
    return crop_to_plot_area(img, contract.plot_bbox_px), contract


def test_dual_resolution_traces_are_identical_normalized():
    """The core no-constants proof: 1x and 2x renders must yield the same
    normalized traces. A leaked pixel literal would shift the 2x result."""
    t1 = _trace_normalized(1)
    t2 = _trace_normalized(2)
    assert set(t1) == set(t2) == {"chg", "dis"}
    for sid in ("chg", "dis"):
        diff = np.abs(t1[sid] - t2[sid])
        assert diff.max() < 0.02, (
            f"{sid}: max normalized-y difference {diff.max():.3f} between "
            f"1x and 2x -- a resolution-dependent constant leaked"
        )


def test_colored_annotation_masks_label_and_rule_but_spares_curve():
    from voltage_tracer.metrics import colored_annotation_mask
    red = (220, 20, 20)
    img = np.full((200, 320, 3), 245, np.uint8)  # pale background
    cv2.line(img, (10, 185), (300, 20), red, 2)   # a curve: one big component
    cv2.line(img, (150, 150), (250, 150), red, 2)  # a horizontal rule/arrow
    for gx in (110, 123, 136):                     # 3 glyph-like blobs = a label
        img[95:114, gx:gx + 9] = red
    mask = colored_annotation_mask(img, [red], 45.0, 2.0, DEFAULTS)
    assert mask[150, 200], "horizontal rule/arrow not masked"
    assert mask[104, 114], "text label glyphs not masked"
    # the curve (a large component) must be spared, even near the annotations
    assert not mask[30, 285], "curve body was masked"
    assert not mask[176, 20], "curve body was masked"


def test_colored_annotation_empty_without_annotations():
    from voltage_tracer.metrics import colored_annotation_mask
    red = (220, 20, 20)
    img = np.full((200, 320, 3), 245, np.uint8)
    cv2.line(img, (10, 185), (300, 20), red, 2)   # only a curve, no labels/rules
    assert not colored_annotation_mask(img, [red], 45.0, 2.0, DEFAULTS).any()


def test_text_region_detected_and_curve_not_flagged():
    img, contract = _make_figure(2)
    crop = crop_to_plot_area(img, contract.plot_bbox_px)
    regions = detect_text_regions(crop, DEFAULTS)
    assert regions, "title text not detected"
    # every detected region sits in the top band (where the title is),
    # never over the curve body lower down
    h = crop.shape[0]
    for x0, y0, x1, y1 in regions:
        assert y1 < 0.35 * h, f"text region {(x0, y0, x1, y1)} overlaps curve body"


def test_curve_only_figure_has_no_false_text_regions():
    """A figure with no text must yield no text regions -- the detector
    must not mistake curves (even a lone short segment) for glyph clusters."""
    img, contract = _make_figure(2)
    # paint over the title with white
    crop = crop_to_plot_area(img, contract.plot_bbox_px)
    crop[: int(0.2 * crop.shape[0]), :] = 255
    assert detect_text_regions(crop, DEFAULTS) == []


def test_text_region_encloses_thin_punctuation():
    """A label with thin strokes that fail the per-glyph size test (an
    equals sign, parentheses) must still be fully enclosed -- otherwise the
    leftover glyph pixels just outside the box become phantom curve points."""
    # a tall crop so the ~28px text is well under glyph_max_hfrac of height
    crop = np.full((420, 320, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "X=Cl", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    regions = detect_text_regions(crop, DEFAULTS)
    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    # every black text pixel -- including the thin "=" strokes -- must lie
    # inside the region
    black = (crop.sum(axis=2) < 200)
    ys, xs = np.where(black)
    assert xs.min() >= x0 and xs.max() < x1 and ys.min() >= y0 and ys.max() < y1


def test_swatch_strips_returns_adjacent_swatch_as_its_own_region():
    from voltage_tracer.metrics import swatch_strips

    # text label at x60-100; a flat colored swatch at x25-57 abutting it
    colored = np.zeros((80, 200), dtype=bool)
    colored[40:42, 25:57] = True  # the swatch, on the label's rows
    # a curve fragment elsewhere in the band but NOT adjacent to the label
    colored[38:44, 150:158] = True
    regions = [(60, 30, 100, 52)]
    strips = swatch_strips(regions, colored, line_width=2.0, d=DEFAULTS)
    assert len(strips) == 1
    sx0, sy0, sx1, sy1 = strips[0]
    assert sx0 <= 25 and sx1 == 60, "strip should span the swatch up to the label edge"
    assert not (sx0 <= 150 <= sx1), "strip must not reach the far fragment"


def test_swatch_strips_empty_without_adjacent_color():
    from voltage_tracer.metrics import swatch_strips

    colored = np.zeros((80, 200), dtype=bool)
    assert swatch_strips([(60, 30, 100, 52)], colored, 2.0, DEFAULTS) == []


def test_fragmented_achromatic_curve_makes_no_text_region():
    """A black curve chopped into compact fragments at crossings must not
    be read as a text line -- the fragments spread vertically, unlike a
    real (single-baseline) label."""
    crop = np.full((200, 300, 3), 255, dtype=np.uint8)
    # scatter compact black blobs along a diagonal (a fragmented curve)
    for k in range(8):
        cx, cy = 30 + 14 * k, 30 + 18 * k
        cv2.rectangle(crop, (cx, cy), (cx + 6, cy + 8), (0, 0, 0), -1)
    assert detect_text_regions(crop, DEFAULTS) == []


def _line_pixels(x0, x1, slope, perp_std, n=400, seed=0):
    """A pixel cloud along y = slope*x, jittered perpendicular by perp_std."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x0, x1, n)
    ys = slope * xs + rng.normal(0.0, perp_std, n)
    return np.round(xs).astype(int), np.round(ys).astype(int)


def test_is_thin_diagonal_discriminates_curve_from_text():
    """The thin-diagonal guard mirrors the real P2-fig5a numbers: a curve
    fragment is one thin sloped stroke; text is horizontal or thicker."""
    from voltage_tracer.metrics import _is_thin_diagonal

    lw = 2.0
    # a shallow-diagonal single stroke (measured: slope ~-0.38, perp ~0.86*lw)
    xs, ys = _line_pixels(60, 160, slope=-0.38, perp_std=0.86 * lw)
    assert _is_thin_diagonal(xs, ys, lw, DEFAULTS) is True

    # a horizontal text baseline: near-zero slope -> kept (not diagonal)
    xs, ys = _line_pixels(0, 250, slope=-0.02, perp_std=3.6 * lw)
    assert _is_thin_diagonal(xs, ys, lw, DEFAULTS) is False

    # a thin HORIZONTAL rule (underline): thin but not diagonal -> kept
    xs, ys = _line_pixels(0, 250, slope=0.0, perp_std=0.5 * lw)
    assert _is_thin_diagonal(xs, ys, lw, DEFAULTS) is False

    # a dense X-crossing tangle: diagonal but thick (two directions) -> NOT
    # skipped; this guard intentionally spares only clean thin strokes
    xs, ys = _line_pixels(60, 160, slope=-0.38, perp_std=6.0 * lw)
    assert _is_thin_diagonal(xs, ys, lw, DEFAULTS) is False
