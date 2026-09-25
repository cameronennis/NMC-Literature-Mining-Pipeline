"""Per-figure reference quantities, measured once right after cropping.

`FigureMetrics` is the single source of every "how many pixels" / "how
steep" / "how far in color" number the pipeline needs. Downstream modules
never invent an absolute figure number: they read a measured field here, or
a derived property (a `defaults.Defaults` multiplier times a measured
field). See `defaults.py` for why. This is what lets the same code trace a
500px figure and a 4000px figure without re-tuning.

`compute_metrics(crop, contract)` is called once; the resulting immutable
`FigureMetrics` is threaded through preprocess/skeletonize/tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from voltage_tracer.contract import AxisScale, Contract
from voltage_tracer.defaults import DEFAULTS, Defaults

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in crop-local pixels


def _axis_px_per_unit(ticks_px, ticks_val, scale: AxisScale) -> float:
    """Pixels per data unit for one axis, from a least-squares fit over all
    tick pairs. For a log axis, per-unit is in log10 space (callers doing
    data math handle the transform; this is only a ruler for pixel spans)."""
    vals = np.asarray(ticks_val, dtype=float)
    if scale is AxisScale.log:
        vals = np.log10(vals)
    slope = np.polyfit(vals, np.asarray(ticks_px, dtype=float), 1)[0]
    return abs(float(slope))


def _measure_line_width(masks: list[np.ndarray]) -> float:
    """Typical curve stroke thickness = median vertical run length of the
    color masks, sampled across columns. Median shrugs off both the long
    runs at steep segments/crossings (where strokes merge) and 1px
    anti-alias slivers. Floored at 1."""
    runlens: list[int] = []
    for mask in masks:
        h, w = mask.shape
        step = max(1, w // 200)  # ~200 sampled columns, resolution-independent
        for x in range(0, w, step):
            idx = np.where(mask[:, x])[0]
            if idx.size == 0:
                continue
            for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                runlens.append(run.size)
    if not runlens:
        return 1.0
    return max(1.0, float(np.median(runlens)))


def swatch_strips(
    regions: list[BBox], colored: np.ndarray, line_width: float, d: Defaults
) -> list[BBox]:
    """Return the colored legend-swatch strips adjacent to each text label,
    as their OWN thin regions (separate from the label boxes).

    A legend swatch is a short flat colored line segment sitting right next
    to its label -- it can't be told from a curve by length (a fast-rate
    curve chopped by crossings leaves fragments shorter than a swatch), but
    it is uniquely identifiable by ADJACENCY: it touches a label on the
    label's own rows. Walking outward from the label edge through contiguous
    colored columns (tolerating a `swatch_gap`) reaches exactly the swatch
    and stops at the clear space beyond; a curve elsewhere in the band is
    never touched because it isn't contiguous with the label edge.

    Returned separately (rather than merged into the label box) so the label
    box can be masked by its achromatic INK only -- preserving a colored
    curve that happens to pass through the label -- while the swatch, being
    genuinely spurious, is masked in full."""
    h, w = colored.shape
    max_gap = max(2, int(round(d.swatch_gap_lw * line_width)))
    flat_tol = max(3, int(round(d.swatch_flat_lw * line_width)))
    strips: list[BBox] = []

    def _flat(x_lo, x_hi):
        """A swatch is a THIN FLAT segment: the colored pixels in the strip
        stay within `flat_tol` rows. A curve merely passing through the
        label's band spans most of it -- reject those so an achromatic
        series (whose 'colored' mask IS its own curve) can't have its curve
        walked and masked as a fake swatch."""
        ys, _ = np.where(colored[y0:y1, x_lo:x_hi])
        return ys.size > 0 and (ys.max() - ys.min()) <= flat_tol

    for (x0, y0, x1, y1) in regions:
        col_has = colored[y0:y1].any(axis=0)
        nx0, gap, x = x0, 0, x0
        while x - 1 >= 0:
            x -= 1
            if col_has[x]:
                nx0, gap = x, 0
            elif (gap := gap + 1) > max_gap:
                break
        if nx0 < x0 and _flat(nx0, x0):
            strips.append((nx0, y0, x0, y1))
        nx1, gap, x = x1, 0, x1 - 1
        while x + 1 < w:
            x += 1
            if col_has[x]:
                nx1, gap = x + 1, 0
            elif (gap := gap + 1) > max_gap:
                break
        if nx1 > x1 and _flat(x1, nx1):
            strips.append((x1, y0, nx1, y1))
    return strips


def _pairwise_color_sep(colors: list[tuple[int, int, int]]) -> float:
    """Minimum pairwise Lab distance between the distinct declared series
    colors (encoded-Lab units). `inf` when fewer than two distinct colors
    exist (no neighbour to bleed into -- callers cap on an absolute)."""
    if len(colors) < 2:
        return float("inf")
    labs = [
        cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2LAB).astype(float)[0, 0]
        for c in colors
    ]
    return min(
        float(np.linalg.norm(labs[i] - labs[j]))
        for i in range(len(labs))
        for j in range(i + 1, len(labs))
    )


def _is_thin_diagonal(xs: np.ndarray, ys: np.ndarray, line_width: float, d: Defaults) -> bool:
    """True if the ink at (xs, ys) is one THIN DIAGONAL stroke, not text.

    A curve fragment chopped at a crossing passes the glyph-size and
    row-span gates but, unlike a real text cluster, its pixels lie along a
    single sloped line: the principal axis is diagonal (|slope| high) and the
    perpendicular spread is only about one stroke thick. Real in-plot text is
    near-horizontal (a baseline, |slope| ~ 0) OR clearly thicker than a stroke
    (stacked glyph strokes), so it fails one of the two conditions and is kept.
    Boxing out a diagonal curve segment interrupts the curve; sparing it costs
    at most a short spurious track the tracker rejects on its own."""
    if xs.size < 3:
        return False
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    evals, evecs = np.linalg.eigh((pts.T @ pts) / len(pts))  # ascending
    major = evecs[:, 1]
    perp_std = float(np.sqrt(max(evals[0], 0.0)))
    slope = abs(major[1] / major[0]) if abs(major[0]) > 1e-9 else float("inf")
    thin = perp_std <= d.text_line_thin_lw * max(line_width, 1.0)
    diagonal = slope >= d.text_line_diag_min_slope
    return bool(thin and diagonal)


def detect_text_regions(
    crop: np.ndarray, d: Defaults, line_width: float | None = None
) -> list[BBox]:
    """Auto-detect text/label regions inside the plot: tight bounding boxes
    around each cluster of achromatic text strokes (a title, a panel label
    like "(a)" or "X=Cl", an in-plot annotation).

    Two-stage, so the region encloses ALL of the text -- including thin
    strokes (`=`, parentheses, punctuation) that fail a per-glyph size test
    and, if left out, leave phantom pixels just outside the box that a curve
    then traces:

    1. STRONG glyphs -- achromatic components of glyph-like height/width/
       fill -- locate where text is. A curve fragment rarely looks like a
       glyph, and a whole curve is one long low-fill component that fails
       outright, so this stage does not fire on curves.
    2. The achromatic mask is DILATED horizontally (by ~a glyph gap) to fuse
       each label's strokes -- strong glyphs and thin punctuation alike --
       into one connected blob. Any blob containing at least
       `min_glyphs_per_cluster` strong glyphs is emitted as the bounding box
       of the *original* achromatic pixels inside it, padded. Because the
       expansion happens only within achromatic pixels, it can never swallow
       a (chromatic) curve -- only the text's own faint/thin strokes.

    All thresholds are fractions of plot dimensions or stroke width (see
    `Defaults`), so this scales with figure resolution."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(float)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    achromatic = (chroma < d.text_chroma_max) & (lab[..., 0] < d.text_dark_lfrac * 255)
    # Reference stroke width for the thin-diagonal test (the caller passes the
    # measured curve stroke width; fall back to the glyph ink's own width).
    lw = line_width if line_width is not None else _measure_line_width([achromatic])

    return _glyph_clusters(achromatic, h, w, lw, d)


def colored_annotation_mask(
    crop: np.ndarray, series_colors: list[tuple[int, int, int]],
    color_tol: float, lw: float, d: Defaults
) -> np.ndarray:
    """Boolean mask of COLOURED annotations that share a series colour and a
    curve would otherwise trace: text labels (a red "79.0%") and thin
    horizontal rules/arrows. Worked PER colour so differently-coloured dashes
    never fuse, and only on SMALL isolated components -- a curve, or its steep
    terminal drop, is one large connected component and is always spared."""
    from voltage_tracer.preprocess import color_mask  # local: avoid import cycle

    h, w = crop.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    big_w, big_h = d.annot_curve_span_frac * w, d.annot_curve_span_frac * h
    rule_min_w, rule_max_h = d.annot_rule_min_wfrac * w, d.annot_rule_max_lw * lw
    for c in series_colors:
        cm = color_mask(crop, c, tolerance=color_tol, hue_recover=True)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm.astype(np.uint8), connectivity=8)
        small = np.zeros((h, w), dtype=bool)  # everything but curve-sized blobs
        for i in range(1, num):
            _, _, bw, bh, _ = stats[i]
            if bw >= big_w or bh >= big_h:
                continue  # a curve / its drop -- never mask
            small |= labels == i
            # a long, thin, horizontal rule or arrow (isolated from the curve)
            if bw >= rule_min_w and bh <= rule_max_h:
                out |= labels == i
        # glyph-cluster label regions: mask the small components inside them
        # (the large curve is excluded from `small`, so it is spared even when
        # a label's bounding box overlaps it)
        for x1, y1, x2, y2 in _glyph_clusters(cm, h, w, lw, d):
            box = np.zeros((h, w), dtype=bool)
            box[y1:y2, x1:x2] = True
            out |= small & box
    if out.any():
        out = cv2.dilate(out.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))).astype(bool)
    return out


def _glyph_clusters(
    glyph_px: np.ndarray, h: int, w: int, lw: float, d: Defaults
) -> list[BBox]:
    """Cluster an ink mask into text-label bounding boxes (the two-stage logic
    shared by the achromatic and coloured passes of `detect_text_regions`)."""
    # stage 1: scan components once. Keep small components as the
    # text-stroke mask (glyphs AND thin punctuation like `=` / parens), and
    # collect STRONG glyphs -- glyph-sized, well-filled -- as (x, y, h)
    # anchors that locate real text lines. A whole curve is one huge
    # component (dropped here); a curve chopped into small fragments at
    # crossings survives into text_stroke, but is excluded below by the
    # requirement that a region sit on a strong-glyph baseline.
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        glyph_px.astype(np.uint8), connectivity=8
    )
    text_stroke = np.zeros((h, w), dtype=np.uint8)
    strong: list[tuple[float, float, int]] = []  # (cx, cy, glyph_height)
    for i in range(1, num):
        _, _, bw, bh, area = stats[i]
        if bh > d.glyph_max_hfrac * h or bw > d.glyph_max_wfrac * w:
            continue
        text_stroke[labels == i] = 1
        if d.glyph_min_hfrac * h <= bh and area / (bw * bh) >= d.glyph_min_extent:
            strong.append((cents[i][0], cents[i][1], int(bh)))
    if len(strong) < d.min_glyphs_per_cluster:
        return []

    # stage 2: fuse each label's strokes with a HORIZONTAL-ONLY dilation
    # (kernel height 1). Letters/punctuation of one label share rows and
    # merge; a curve crossing above or below the text sits on different
    # rows and stays a separate blob, so the region can never grow
    # vertically along a curve. Keep blobs holding >= min_glyphs strong
    # glyphs; a fragmented curve makes blobs with none.
    kx = max(1, int(round(d.glyph_gap_wfrac * w)))
    pad = max(1, int(round(d.text_pad_lw * max(1.0, _measure_line_width([glyph_px])))))
    dilated = cv2.dilate(text_stroke,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (2 * kx + 1, 1)))
    dnum, dlabels, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    blob_glyphs: dict[int, list[int]] = {}
    for gx, gy, gh in strong:
        lbl = int(dlabels[int(gy), int(gx)])
        if lbl:
            blob_glyphs.setdefault(lbl, []).append(gh)

    regions: list[BBox] = []
    for lbl, heights in blob_glyphs.items():
        if len(heights) < d.min_glyphs_per_cluster:
            continue
        ys, xs = np.where((dlabels == lbl) & text_stroke.astype(bool))
        if xs.size == 0:
            continue
        # a real text line's strokes span about one glyph height; a blob of
        # curve fragments (achromatic pieces of a black curve chopped at
        # crossings) spreads across many rows -- reject those.
        if (ys.max() - ys.min()) > d.text_line_height_k * max(heights):
            continue
        # ...and a SHALLOW-diagonal curve fragment spans few rows but is one
        # thin sloped stroke, not a text cluster -- spare it too.
        if _is_thin_diagonal(xs, ys, lw, d):
            continue
        regions.append((
            max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad),
        ))
    return regions


@dataclass(frozen=True)
class FigureMetrics:
    plot_w_px: int
    plot_h_px: int
    line_width_px: float
    px_per_xunit: float
    px_per_yunit: float
    intercurve_color_sep: float
    text_regions: list[BBox] = field(default_factory=list)   # label boxes (ink-masked)
    swatch_regions: list[BBox] = field(default_factory=list)  # legend swatches (full-masked)
    d: Defaults = DEFAULTS

    # --- color tolerances (encoded-Lab units) ---
    @property
    def color_tol(self) -> float:
        return min(self.d.color_tol_sepfrac * self.intercurve_color_sep,
                   self.d.color_tol_abs_cap)

    @property
    def lenient_tol(self) -> float:
        return min(self.d.lenient_tol_sepfrac * self.intercurve_color_sep,
                   self.d.lenient_tol_abs_cap)

    # --- pixel-area / thickness derived counts ---
    @property
    def speckle_min_area(self) -> int:
        return max(2, int(round(self.d.speckle_area_k * self.line_width_px ** 2)))

    @property
    def merge_gap(self) -> int:
        return max(1, int(round(self.d.merge_gap_lw * self.line_width_px)))

    # --- horizontal spans (columns) ---
    def _cols(self, wfrac: float, floor: int = 1) -> int:
        return max(floor, int(round(wfrac * self.plot_w_px)))

    @property
    def slope_window_cols(self) -> int:
        return self._cols(self.d.slope_window_wfrac, floor=2)

    @property
    def init_columns(self) -> int:
        return self._cols(self.d.init_wfrac, floor=3)

    @property
    def max_link_gap_cols(self) -> int:
        return self._cols(self.d.link_gap_wfrac, floor=1)

    @property
    def confirm_run_cols(self) -> int:
        return self._cols(self.d.confirm_run_wfrac, floor=3)

    @property
    def deactivate_cols(self) -> int:
        return self._cols(self.d.deactivate_wfrac, floor=5)

    @property
    def trim_window_cols(self) -> int:
        return self._cols(self.d.trim_window_wfrac, floor=5)

    @property
    def head_probe_cols(self) -> int:
        return self._cols(self.d.head_probe_wfrac, floor=4)

    @property
    def pending_gap_cols(self) -> int:
        return self._cols(self.d.pending_gap_wfrac, floor=1)

    @property
    def bridge_min_gap_cols(self) -> int:
        return self._cols(self.d.bridge_min_gap_wfrac, floor=2)

    @property
    def bridge_min_chain_cols(self) -> int:
        return self._cols(self.d.bridge_min_chain_wfrac, floor=3)

    @property
    def flat_min_cols(self) -> int:
        return self._cols(self.d.flat_min_wfrac, floor=5)

    @property
    def coherence_min_span_cols(self) -> int:
        return self._cols(self.d.coherence_min_span_wfrac, floor=10)

    @property
    def coast_max_cols(self) -> int:
        return self._cols(self.d.coast_max_wfrac, floor=8)

    # --- vertical tolerances (pixels) ---
    @property
    def max_jump_px(self) -> float:
        return self.d.max_jump_hfrac * self.plot_h_px

    @property
    def reversal_tol_px(self) -> float:
        return self.d.reversal_tol_hfrac * self.plot_h_px

    @property
    def link_tol_px(self) -> float:
        return self.d.link_tol_lw * self.line_width_px

    @property
    def head_residual_px(self) -> float:
        return self.d.head_residual_hfrac * self.plot_h_px

    # --- terminal-drop extension (pixels / columns) ---
    @property
    def ext_col_offset(self) -> int:
        return max(1, int(round(self.d.ext_col_offset_lw * self.line_width_px)))

    @property
    def ext_run_gap_px(self) -> float:
        return self.d.ext_run_gap_hfrac * self.plot_h_px

    @property
    def ext_min_extension_px(self) -> float:
        return self.d.ext_min_extension_hfrac * self.plot_h_px

    @property
    def ext_step_px(self) -> float:
        return max(1.0, self.d.ext_step_lw * self.line_width_px)

    @property
    def ext_min_advance_px(self) -> float:
        return self.d.ext_min_advance_lw * self.line_width_px

    @property
    def ext_back_overlap_px(self) -> float:
        return self.d.ext_back_overlap_lw * self.line_width_px


def compute_metrics(
    crop: np.ndarray, contract: Contract, d: Defaults = DEFAULTS
) -> FigureMetrics:
    """Measure all per-figure reference quantities from the cropped plot."""
    from voltage_tracer.preprocess import color_mask  # local: avoid cycle

    h, w = crop.shape[:2]
    colors = sorted({s.color_rgb for s in contract.series})
    sep = _pairwise_color_sep(colors)
    strict_tol = min(d.color_tol_sepfrac * sep, d.color_tol_abs_cap)
    masks = [color_mask(crop, c, tolerance=strict_tol) for c in colors]
    line_width = _measure_line_width(masks)

    text_regions = detect_text_regions(crop, d, line_width=line_width)
    colored = np.zeros((h, w), dtype=bool)
    for mk in masks:
        colored |= mk
    swatches = swatch_strips(text_regions, colored, line_width, d)

    return FigureMetrics(
        plot_w_px=w,
        plot_h_px=h,
        line_width_px=line_width,
        px_per_xunit=_axis_px_per_unit(
            contract.x_axis.ticks_px, contract.x_axis.ticks_val, contract.x_axis.scale
        ),
        px_per_yunit=_axis_px_per_unit(
            contract.y_axis.ticks_px, contract.y_axis.ticks_val, contract.y_axis.scale
        ),
        intercurve_color_sep=sep,
        text_regions=text_regions,
        swatch_regions=swatches,
        d=d,
    )
