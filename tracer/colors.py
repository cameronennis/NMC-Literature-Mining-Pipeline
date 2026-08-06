"""Identify distinct line colors inside a plot interior and build a clean
per-line pixel mask for each one (legend-text blobs removed)."""
import cv2
import numpy as np

# Hue centers (OpenCV 0-179 scale) used only to give traces human-readable names.
_HUE_NAMES = [
    (0, "red"), (10, "orange"), (25, "yellow"), (45, "yellow-green"),
    (65, "green"), (90, "cyan"), (120, "blue"), (140, "purple"),
    (155, "magenta"), (170, "red"), (180, "red"),
]


def _hue_to_name(hue):
    best = min(_HUE_NAMES, key=lambda hn: min(abs(hue - hn[0]), 180 - abs(hue - hn[0])))
    return best[1]


def _split_text_blobs(mask, max_extent=0.25, min_area=10):
    """Split connected components into (curve_like, text_like) masks using
    bounding-box fill ratio ("extent"): a curve stroke's extent is near-zero
    no matter its slope, while a text glyph's is high (0.3-1.0). Both masks
    are returned -- legend_ocr.py reads text_like instead of discarding it,
    since it's exactly the legend/tick-label text this step would otherwise
    throw away."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    curve_like = np.zeros_like(mask, dtype=bool)
    text_like = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        extent = area / (w * h)
        if extent <= max_extent:
            curve_like |= labels == i
        else:
            text_like |= labels == i
    return curve_like, text_like


def find_line_masks(interior_bgr, sat_thresh=60, dark_val_thresh=120,
                     min_area_frac=0.003, hue_bin=8, return_rejected=False):
    """Return a list of {"name": str, "color_bgr": (b,g,r), "mask": bool array}
    for each distinct line color found in `interior_bgr` (a crop that
    excludes the border itself). Two lines that happen to share almost the
    exact same hue (e.g. a medium blue and navy) will come back as one mask
    -- extract.mask_to_paths still recovers each curve separately since it
    tracks strokes by position/continuity, not color, it just can't give
    them different color-based names.

    If `return_rejected=True`, also returns the combined mask of every
    text-like blob rejected along the way (legend labels, tick labels), for
    legend_ocr.py to OCR instead of it being silently discarded."""
    h, w = interior_bgr.shape[:2]
    area = h * w
    hsv = cv2.cvtColor(interior_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Saturation alone separates colored line pixels from white/black/gray
    # (background, border, text, and the achromatic black line) regardless
    # of how bright the color is.
    chromatic = sat > sat_thresh
    achromatic_dark = (sat <= sat_thresh) & (val < dark_val_thresh)

    results = []
    rejected = np.zeros((h, w), dtype=bool)

    # --- chromatic (colored) lines: find hue peaks, then mask each one ---
    hist, edges = np.histogram(hue[chromatic], bins=180 // hue_bin, range=(0, 180))
    min_count = min_area_frac * area
    used = np.zeros_like(chromatic)
    order = np.argsort(hist)[::-1]
    for i in order:
        if hist[i] < min_count:
            continue
        center = (edges[i] + edges[i + 1]) / 2
        band = (np.minimum(np.abs(hue - center), 180 - np.abs(hue - center)) < hue_bin) & chromatic
        band = band & ~used
        if band.sum() < min_count:
            continue
        clean, text_like = _split_text_blobs(band)
        rejected |= text_like
        if clean.sum() < min_count:
            continue
        used |= band
        mean_bgr = interior_bgr[clean].mean(axis=0).astype(int)
        results.append({
            "name": _hue_to_name(center),
            "color_bgr": tuple(int(c) for c in mean_bgr),
            "mask": clean,
        })

    # --- achromatic (black/dark gray) line ---
    clean_black, text_like_black = _split_text_blobs(achromatic_dark)
    rejected |= text_like_black
    if clean_black.sum() >= min_count:
        mean_bgr = interior_bgr[clean_black].mean(axis=0).astype(int)
        results.append({
            "name": "black",
            "color_bgr": tuple(int(c) for c in mean_bgr),
            "mask": clean_black,
        })

    return (results, rejected) if return_rejected else results
