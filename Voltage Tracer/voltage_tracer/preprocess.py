"""Image loading, plot-area cropping, and pixel masking.

All functions work in RGB (matching the contract's `color_rgb`), not
OpenCV's default BGR -- `load_image` does the conversion once at the edge.
Masks returned by this module are boolean arrays where True marks pixels
to *exclude* (gridlines, axes, legend) or, for `color_mask`, pixels that
*belong* to the requested series color.
"""
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array. Fails loudly if the file
    is missing or unreadable (cv2.imread returns None silently otherwise)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image file: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_to_plot_area(
    image: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray:
    """Crop to the contract's plot_bbox_px ([x0, y0, x1, y1], original-image
    pixels), clamped to the image bounds. Downstream pixel coordinates are
    all relative to this crop; calibrate.py maps them back through the
    contract's tick pixel positions, which use the same original-image frame,
    so callers must offset by (x0, y0) -- pipeline.py owns that bookkeeping."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"plot bbox {bbox} does not overlap the {w}x{h} image"
        )
    return image[y0:y1, x0:x1]


def mask_outside_data_region(
    shape: tuple[int, int],
    region_frac: tuple[float, float, float, float],
) -> np.ndarray:
    """Boolean mask (True = pixel OUTSIDE the data region, to exclude) for
    a crop of the given (height, width).

    `region_frac` is the contract's `data_region_frac`
    ([fx0, fy0, fx1, fy1] fractions of the crop, fy measured from the top
    row). Excludes the complementary border bands -- an in-plot title, a
    panel letter, or a legend over the axes -- so their text pixels never
    reach color masking and terminal extensions can't run into them. A
    full-box region (the default) masks nothing."""
    h, w = shape
    mask = np.ones((h, w), dtype=bool)
    fx0, fy0, fx1, fy1 = region_frac
    x0, x1 = int(round(fx0 * w)), int(round(fx1 * w))
    y0, y1 = int(round(fy0 * h)), int(round(fy1 * h))
    mask[y0:y1, x0:x1] = False
    return mask


def mask_regions(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Boolean mask (True = inside one of `boxes`) for a crop of the given
    (height, width). `boxes` are (x0, y0, x1, y1) in crop-local pixels --
    used to exclude auto-detected `FigureMetrics.text_regions` (and any
    contract override regions) before color masking."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def mask_text_ink(
    crop: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    chroma_max: float = 22.0,
    dark_lfrac: float = 0.6,
    dilate_px: int = 2,
) -> np.ndarray:
    """Boolean mask of the achromatic text INK inside each label box (not the
    whole box), dilated by `dilate_px`.

    Masking only the near-neutral dark glyph pixels -- not the full rectangle
    -- means a COLORED curve passing through a label or title (a charge
    plateau under a centered title, a discharge fanning through an in-plot
    legend) survives, while the text itself is removed. The tracker then
    bridges only the few pixels where the curve runs directly behind a
    glyph. Chromatic curves are left untouched because the ink test is on
    low Lab chroma."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(float)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ink = (chroma < chroma_max) & (lab[..., 0] < dark_lfrac * 255)
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] |= ink[y0:y1, x0:x1]
    if dilate_px > 0 and mask.any():
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(bool)
    return mask


def mask_gridlines_and_axes(
    image: np.ndarray,
    angle_tolerance_deg: float = 2.0,
    min_length_frac: float = 0.85,
    line_thickness: int = 3,
    max_line_gap: int = 15,
    sat_thresh: int = 60,
    border_span_frac: float = 0.9,
    dark_thresh: int = 110,
) -> np.ndarray:
    """Boolean mask (True = pixel belongs to a gridline/axis line).

    Two complementary detectors, both restricted to achromatic
    (low-saturation) pixels so a *colored* curve's plateau is never eaten:

    1. HoughLinesP for long near-horizontal/near-vertical segments.
       `min_length_frac` (fraction of the dimension a segment runs along)
       must stay high: real gridlines/axis borders span essentially the
       whole plot, whereas even a black curve's flat plateau rarely exceeds
       ~2/3 of it. `max_line_gap` is generous (a skewed 1px line rasterizes
       as a staircase Hough would otherwise fragment).
    2. A direct near-full-span dark-fraction pass: any row/column that is
       dark across at least `border_span_frac` of its length is a solid
       axis border, masked in full. This catches a border sitting exactly
       at the crop edge (where the plot box is cropped to its own border),
       which Canny/Hough miss for lack of an off-edge gradient -- while
       still sparing plateaus, which never span that fraction."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    achromatic = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1] <= sat_thresh
    edges = cv2.Canny(gray, 50, 150)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_length_frac * min(h, w),
        maxLineGap=max_line_gap,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if segments is not None:
        for x0, y0, x1, y1 in segments.reshape(-1, 4):
            angle = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 180
            near_horizontal = angle <= angle_tolerance_deg or angle >= 180 - angle_tolerance_deg
            near_vertical = abs(angle - 90) <= angle_tolerance_deg
            length = float(np.hypot(x1 - x0, y1 - y0))
            if near_horizontal and length >= min_length_frac * w:
                cv2.line(mask, (x0, y0), (x1, y1), 1, thickness=line_thickness)
            elif near_vertical and length >= min_length_frac * h:
                cv2.line(mask, (x0, y0), (x1, y1), 1, thickness=line_thickness)
    out = mask.astype(bool) & achromatic

    # direct border pass: full-span achromatic-dark rows/columns
    dark = achromatic & (gray < dark_thresh)
    t = max(1, line_thickness // 2)
    for y in np.where(dark.mean(axis=1) >= border_span_frac)[0]:
        out[max(0, y - t):y + t + 1, :] = True
    for x in np.where(dark.mean(axis=0) >= border_span_frac)[0]:
        out[:, max(0, x - t):x + t + 1] = True
    return out


def mask_legend_region(
    image: np.ndarray,
    override_bbox: tuple[float, float, float, float] | None = None,
    sat_thresh: int = 60,
    max_blob_frac: float = 0.005,
    max_blob_dim_frac: float = 0.15,
    min_extent: float = 0.3,
    text_search_px: int = 30,
    pad_px: int = 10,
    curve_spare_hfrac: float = 0.5,
) -> np.ndarray:
    """Boolean mask (True = legend region to exclude).

    If the contract supplies `legend_bbox_px`, pass it as `override_bbox`: that
    rectangle is masked, EXCEPT a colored curve that merely transits it. A
    reasoning-stage legend box is often drawn too large and clips a curve (e.g.
    a discharge terminal drop passing under the legend); full-masking would
    erase that drop. So a saturated stroke spanning at least `curve_spare_hfrac`
    of the box height -- a curve passing through, not a short colour swatch or
    achromatic text -- is spared, mirroring how `mask_text_ink` keeps a colored
    curve that runs through a text label. Otherwise a simple heuristic: legend
    swatches are small, compact, isolated blobs of saturated color sitting next
    to dark text-like pixels, whereas data curves are long low-"extent" strokes.

    Deliberately conservative: masking nothing (all-False) is a safe
    failure mode, since the tracker just sees a legend swatch as a short
    unconfirmed track; masking part of a real curve is not.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    if override_bbox is not None:
        x0, y0, x1, y1 = override_bbox
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
        if x1 <= x0 or y1 <= y0:
            return mask
        mask[y0:y1, x0:x1] = True
        # spare a curve transiting the box: a saturated component tall enough to
        # span the box is a passing-through stroke, not a swatch or text.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        sat = (hsv[y0:y1, x0:x1, 1] > sat_thresh) & (hsv[y0:y1, x0:x1, 2] > sat_thresh)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            sat.astype(np.uint8), connectivity=8
        )
        spare = np.zeros_like(sat, dtype=bool)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_HEIGHT] >= curve_spare_hfrac * (y1 - y0):
                spare |= labels == i
        if spare.any():
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            spare = cv2.dilate(spare.astype(np.uint8), k).astype(bool)
            mask[y0:y1, x0:x1][spare] = False
        return mask

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturated = (hsv[..., 1] > sat_thresh) & (hsv[..., 2] > sat_thresh)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark = gray < 100

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        saturated.astype(np.uint8), connectivity=8
    )
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area > max_blob_frac * h * w:
            continue
        if bw > max_blob_dim_frac * w or bh > max_blob_dim_frac * h:
            continue
        if area / (bw * bh) < min_extent:
            continue
        # require adjacent text-like dark pixels (to the right, where legend
        # labels conventionally sit, or immediately around the swatch)
        ny0, ny1 = max(0, y - pad_px), min(h, y + bh + pad_px)
        nx0, nx1 = max(0, x - pad_px), min(w, x + bw + text_search_px)
        neighborhood = dark[ny0:ny1, nx0:nx1].copy()
        # don't let the swatch's own dark-ish pixels count as "text"
        neighborhood[
            max(0, y - ny0) : y - ny0 + bh, max(0, x - nx0) : x - nx0 + bw
        ] = False
        if neighborhood.sum() < 5:
            continue
        mask[ny0:ny1, nx0:nx1] = True
    return mask


def remove_small_components(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """Drop connected components smaller than `min_area` pixels from a
    boolean mask. Color masks on JPEG-compressed figures pick up scattered
    compression-halo speckles (1-4 px) around other curves and text.

    Keep `min_area` low: where several curves cross, a real curve gets
    chopped into short segments only ~10-25 px each, and an over-eager
    floor deletes those genuine fragments (which then read as a missing
    stretch of curve). The tracker's continuity/direction logic is what
    rejects surviving speckle, so this only needs to clear the very
    smallest specks. Run it between `color_mask` and skeletonization."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == i
    return keep


def color_mask(
    image: np.ndarray,
    color_rgb: tuple[int, int, int],
    tolerance: float = 25.0,
    achromatic_chroma: float = 20.0,
    hue_recover: bool = False,
    hue_tol_deg: float = 40.0,
    hue_chroma_min: float = 10.0,
    hue_lightness_margin: float = 20.0,
) -> np.ndarray:
    """Boolean mask of pixels matching `color_rgb`.

    Chromatic targets: pixels within `tolerance` Euclidean distance in Lab
    space -- Lab handles anti-aliased edge pixels (which drift toward the
    background in lightness but keep their chroma) far better than RGB
    thresholding. OpenCV's uint8 Lab encoding scales L to 0-255 and offsets
    a/b by 128; `tolerance` is in those encoded units (a visually-distinct
    color difference is roughly 30+, anti-aliasing drift roughly 10-25).

    Near-achromatic targets (black/gray series -- Lab chroma below
    `achromatic_chroma`): a fixed distance sphere fails here, because JPEG
    compression smears a black curve's lightness across a wide range while
    leaving it colorless. Dispatch instead on the achromatic criterion:
    low chroma AND lightness within a generous band of the target
    (3x tolerance, one-sided toward the background).

    `hue_recover` (chromatic targets only): a distance sphere keyed off a
    bright, saturated reference also misses a curve's OWN pixels where anti-
    aliasing/JPEG darkens and desaturates it -- a bunched low-capacity IR
    drop, a thin crossing -- even though those pixels keep the target's HUE.
    When set, union in pixels whose Lab hue angle is within `hue_tol_deg` of
    the target's, that are colored enough for a reliable hue
    (`hue_chroma_min`), and that are darker than the near-white background by
    `hue_lightness_margin` (so a pale tinted background is never swept in).
    Distinct-hued series (the norm) stay separable; this never widens toward
    a neighbour's hue. Off by default: enabled for the strict tracking mask,
    not for line-width measurement or the already-generous lenient mask."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(
        np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB
    ).astype(np.float32)[0, 0]

    target_chroma = float(np.hypot(target[1] - 128, target[2] - 128))
    if target_chroma < achromatic_chroma:
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        low_side = target[0] - tolerance
        high_side = target[0] + 3 * tolerance
        return (chroma <= achromatic_chroma) & (lab[..., 0] >= low_side) & (
            lab[..., 0] <= high_side
        )

    dist = np.linalg.norm(lab - target, axis=-1)
    mask = dist <= tolerance
    if hue_recover:
        a, b = lab[..., 1] - 128, lab[..., 2] - 128
        chroma = np.hypot(a, b)
        px_hue = np.degrees(np.arctan2(b, a))
        t_hue = np.degrees(np.arctan2(target[2] - 128, target[1] - 128))
        d_hue = np.abs((px_hue - t_hue + 180.0) % 360.0 - 180.0)
        white_l = float(np.percentile(lab[..., 0], 92))
        hue_ok = (
            (d_hue <= hue_tol_deg)
            & (chroma >= hue_chroma_min)
            & (lab[..., 0] <= white_l - hue_lightness_margin)
        )
        mask = mask | hue_ok
    return mask
