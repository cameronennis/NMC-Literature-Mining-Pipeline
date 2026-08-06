"""Read an in-plot legend (color swatch + label text) and match each label
to the curve it names, instead of a hand-typed color->condition mapping."""
import re

import cv2
import numpy as np
import pytesseract

_LABEL_RE = re.compile(r"^-?\d+\.?\d*\s*[a-zA-Z/%\-]*$")
_SAT_THRESH = 60


def _group_label_boxes(rejected_mask, dilate_w=7, dilate_h=4, min_area=15):
    """Merge nearby rejected-text-blob pixels (individual characters) into
    one bounding box per label/word via morphological dilation. `dilate_w`
    is deliberately tuned tight -- just enough to bridge the small gap
    between digits/decimal-point within one number, not wide enough to also
    bridge the larger gap to a neighboring legend entry's swatch or label
    (both would otherwise merge two entries into one unreadable OCR blob)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_w, dilate_h))
    dilated = cv2.dilate(rejected_mask.astype(np.uint8), kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    boxes = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, w, h))
    return boxes


def _swatch_color(interior_bgr, box, sat_thresh=_SAT_THRESH):
    """Look for a short colored dash immediately to the left of a label box
    (the usual "-- 0.2C" legend layout). Returns mean BGR, or None."""
    x, y, w, h = box
    x0 = max(0, x - 40)
    strip = interior_bgr[y:y + h, x0:x]
    if strip.size == 0:
        return None
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    mask = sat > sat_thresh
    if mask.sum() < 3:
        return None
    return tuple(int(c) for c in strip[mask].mean(axis=0))


def _label_text_color(interior_bgr, rejected_mask, box, sat_thresh=_SAT_THRESH):
    """Use the label glyphs' own pixel color if it's saturated (some papers
    color the legend text itself to match its curve, e.g. GraphEx1.png)."""
    x, y, w, h = box
    sub_mask = rejected_mask[y:y + h, x:x + w]
    sub_bgr = interior_bgr[y:y + h, x:x + w]
    if sub_mask.sum() == 0:
        return None
    pixels = sub_bgr[sub_mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat_mask = hsv[:, 1] > sat_thresh
    if sat_mask.sum() < max(3, 0.3 * len(pixels)):
        return None
    return tuple(int(c) for c in pixels[sat_mask].mean(axis=0))


def _lab_distance(bgr_a, bgr_b):
    a = cv2.cvtColor(np.uint8([[bgr_a]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    b = cv2.cvtColor(np.uint8([[bgr_b]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return float(np.linalg.norm(a - b))


def _nearest_curve_by_color(color_bgr, line_masks):
    if not line_masks or color_bgr is None:
        return None, None
    dists = [_lab_distance(color_bgr, lm["color_bgr"]) for lm in line_masks]
    i = int(np.argmin(dists))
    return line_masks[i]["name"], dists[i]


def _nearest_curve_by_proximity(box, line_masks):
    if not line_masks:
        return None
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    best_name, best_dist = None, None
    for lm in line_masks:
        ys, xs = np.where(lm["mask"])
        if len(xs) == 0:
            continue
        d = np.min((xs - cx) ** 2 + (ys - cy) ** 2)
        if best_dist is None or d < best_dist:
            best_dist, best_name = d, lm["name"]
    return best_name


def find_legend_labels(interior_bgr, rejected_mask, line_masks):
    """Return a list of {"text": str, "box": (x,y,w,h), "matched_curve":
    str or None, "color_source": "swatch"/"text_color"/"proximity"/None}
    for each legend-like label found among the rejected text blobs."""
    gray = cv2.cvtColor(interior_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for box in _group_label_boxes(rejected_mask):
        x, y, w, h = box
        pad = 3
        crop = gray[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if crop.size == 0:
            continue
        # Legend text at native figure resolution is often only ~10-15px
        # tall; upscaling before OCR is a standard, cheap accuracy win for
        # small text.
        up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        config = "--psm 7 -c tessedit_char_whitelist='0123456789.-/%CcAaVv '"
        data = pytesseract.image_to_data(up, config=config, output_type=pytesseract.Output.DICT)
        words = [(t.strip(), c) for t, c in zip(data["text"], data["conf"]) if t.strip()]
        if not words:
            continue
        text = " ".join(w for w, _ in words)
        confs = [c for _, c in words if c >= 0]
        ocr_confidence = (min(confs) / 100.0) if confs else 0.0
        if not _LABEL_RE.match(text):
            continue

        color_source = None
        color = _swatch_color(interior_bgr, box)
        if color is not None:
            color_source = "swatch"
        else:
            color = _label_text_color(interior_bgr, rejected_mask, box)
            if color is not None:
                color_source = "text_color"

        matched_curve = None
        if color is not None:
            matched_curve, _ = _nearest_curve_by_color(color, line_masks)
        if matched_curve is None:
            matched_curve = _nearest_curve_by_proximity(box, line_masks)
            if matched_curve is not None:
                color_source = "proximity"

        # A swatch dash misread as looking vaguely numeric, a genuinely
        # low-confidence character read, or a bare single digit (rates in
        # this domain are essentially always shown with a decimal, e.g.
        # "0.5"/"1.0" -- a lone "5" is almost certainly a clipped fragment,
        # not the true value) should be flagged rather than trusted. This is
        # small legend text at native figure resolution; errors here are
        # expected, not exceptional.
        digits_only = text.replace(".", "").replace("-", "")
        needs_review = (
            ocr_confidence < 0.75
            or color_source in (None, "proximity")
            or len(digits_only) < 2
        )

        results.append({
            "text": text,
            "box": box,
            "matched_curve": matched_curve,
            "color_source": color_source,
            "ocr_confidence": ocr_confidence,
            "needs_review": needs_review,
        })
    return results
