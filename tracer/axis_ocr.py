"""Read axis calibration (tick values, units) directly off a graph image via
OCR, instead of a hand-filled config.csv row."""
import re

import cv2
import numpy as np
import pytesseract

# Shared with caption_parse.py / metadata_extract.py so unit text extracted
# from an axis title and from body/caption prose normalize to the same
# tokens.
UNIT_VOCAB = [
    "V", "%", "mAh/g", "mAh g-1", "mA cm-2", "mA h cm-2", "mAh/cm2",
    "mAh cm-2", "W/cm2", "mW/cm2", "mg/cm2", "MPa", "degC",
]


def _normalize_unit_text(s):
    s = s.replace("−", "-").replace("⁻", "-").replace("¹", "1").replace("²", "2")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_unit(raw_text, require_match=False):
    """Fuzzy-match OCR'd/caption unit text against the shared vocabulary.
    Returns the canonical vocab string. If nothing matches closely: returns
    the cleaned raw text by default (still visible/reviewable rather than
    dropped -- useful for caption text, which is usually clean), or None
    when `require_match=True` (useful for OCR'd figure text, which can be
    garbled enough that the "cleaned raw text" fallback is just noise)."""
    cleaned = _normalize_unit_text(raw_text)
    cleaned_compact = re.sub(r"[\s\-/]", "", cleaned).lower()
    best, best_score = None, 0.0
    for candidate in UNIT_VOCAB:
        c_compact = re.sub(r"[\s\-/]", "", candidate).lower()
        score = 1.0 if c_compact == cleaned_compact else (
            1.0 if c_compact in cleaned_compact or cleaned_compact in c_compact else 0.0
        )
        if score > best_score:
            best, best_score = candidate, score
    if best_score > 0:
        return best
    return None if require_match else cleaned


def _find_tick_marks(strip_min, dark_thresh=100, merge_gap=2):
    idx = np.where(strip_min < dark_thresh)[0]
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
    return [float(np.mean(g)) for g in groups]


def find_tick_positions(gray, edge, box, near=8, far=2):
    """Scan a thin strip just outside the given border edge for short dark
    tick marks. `box` is (left, right, top, bottom) in this image's pixel
    frame. Returns tick positions in the same frame (row for 'left'/'right'
    edges, column for 'bottom'/'top'). `far` must clear the border line's
    own thickness/anti-aliasing, or every column/row reads as "dark"."""
    left, right, top, bottom = box
    if edge == "left":
        strip = gray[top:bottom + 1, max(0, left - near):max(1, left - far)]
        return [t + top for t in _find_tick_marks(strip.min(axis=1))]
    if edge == "bottom":
        strip = gray[bottom + far:bottom + near, left:right + 1]
        return [t + left for t in _find_tick_marks(strip.min(axis=0))]
    if edge == "top":
        strip = gray[max(0, top - near):max(1, top - far), left:right + 1]
        return [t + left for t in _find_tick_marks(strip.min(axis=0))]
    if edge == "right":
        strip = gray[top:bottom + 1, right + far:right + near]
        return [t + top for t in _find_tick_marks(strip.min(axis=1))]
    raise ValueError(f"unknown edge {edge!r}")


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def read_axis_labels(gray, edge, box, label_size=90, gap=8, tick_line_clearance=20,
                      tick_line_height=30, end_margin=20):
    """OCR every numeric tick label along one axis in a single pass and
    return (pixel_pos, value) pairs using Tesseract's own per-token bounding
    boxes for position -- this sidesteps guessing a fixed crop window per
    tick, which either clips wide labels or bleeds into a neighboring one
    when ticks are closely spaced. `tick_line_clearance` on the bottom edge
    must clear the tick mark's own dash (else OCR reads the dash as a stray
    character glued onto the real number); `tick_line_height` stays tight so
    the strip doesn't also reach the axis title line below it. `end_margin`
    extends past the box's own left/right (or top/bottom) edge, since the
    first/last label is centered on its tick and its text commonly overhangs
    past the plot border -- without this the end labels get clipped and
    misread."""
    left, right, top, bottom = box
    if edge == "left":
        strip = gray[max(0, top - end_margin):bottom + 1 + end_margin,
                      max(0, left - label_size):max(1, left - gap)]
        axis_origin, along = top - end_margin, "row"
    elif edge == "bottom":
        strip = gray[bottom + tick_line_clearance:bottom + tick_line_clearance + tick_line_height,
                      max(0, left - end_margin):right + 1 + end_margin]
        axis_origin, along = left - end_margin, "col"
    else:
        raise ValueError(f"unsupported edge {edge!r} for label reading")
    if strip.size == 0:
        return []

    config = "--psm 6 -c tessedit_char_whitelist='0123456789.- '"
    data = pytesseract.image_to_data(strip, config=config, output_type=pytesseract.Output.DICT)

    pairs = []
    for i, text in enumerate(data["text"]):
        text = text.strip().rstrip("-")  # a trailing '-' is always the tick dash, not a sign
        if not text or not _NUMBER_RE.fullmatch(text):
            continue
        if along == "row":
            center = data["top"][i] + data["height"][i] / 2 + axis_origin
        else:
            center = data["left"][i] + data["width"][i] / 2 + axis_origin
        pairs.append((float(center), float(text)))
    return pairs


def read_axis_unit(gray, box, edge, band=140):
    """OCR the axis title (rotated to horizontal first for a left/right
    edge) and fuzzy-match it against the shared unit vocabulary. Scans a
    generous band since title font size/offset varies by figure, and
    requires a minimum amount of real letter content before trusting the
    result -- an empty/near-blank crop (e.g. a shared axis title that lives
    outside this particular sub-panel) should come back None, not a
    fuzzy-matched guess from OCR noise."""
    left, right, top, bottom = box
    if edge == "left":
        crop = gray[top:bottom + 1, max(0, left - band):max(1, left - 8)]
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif edge == "bottom":
        crop = gray[bottom + 8:bottom + band, left:right + 1]
    else:
        raise ValueError(f"unsupported edge {edge!r} for unit reading")
    if crop.size == 0:
        return None
    text = pytesseract.image_to_string(crop, config="--psm 6").strip()
    letters = sum(c.isalpha() for c in text)
    if letters < 3:
        return None
    # unit is typically the parenthesized trailing part of the title
    m = re.search(r"\(([^)]+)\)", text)
    return match_unit(m.group(1) if m else text, require_match=True)


def calibrate_axis(ticks, max_outlier_frac=0.34):
    """ticks: list of (pixel_pos, value) pairs, >=2 required, >=3 to allow
    outlier rejection. Returns dict with slope/intercept (value = slope*pixel
    + intercept), confidence in [0, 1], and needs_review, or None if
    uncalibratable. Never returns a fit without validating monotonicity."""
    if len(ticks) < 2:
        return None

    pts = sorted(ticks, key=lambda t: t[0])
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    def _fit(xs, ys):
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        resid = ys - pred
        return slope, intercept, resid

    slope, intercept, resid = _fit(xs, ys)

    n_dropped = 0
    max_drops = max(0, len(pts) - len(pts) // 2 - 2)  # never drop below ~half, floor of 2
    while len(xs) >= 4 and n_dropped < max_drops:
        spacing = np.median(np.abs(np.diff(ys))) or 1.0
        worst = np.argmax(np.abs(resid))
        if abs(resid[worst]) <= 0.5 * spacing:
            break
        keep = np.arange(len(xs)) != worst
        xs, ys = xs[keep], ys[keep]
        slope, intercept, resid = _fit(xs, ys)
        n_dropped += 1
    dropped = n_dropped > 0

    diffs = np.diff(ys)
    monotonic = np.all(diffs > 0) or np.all(diffs < 0)

    spacing = np.median(np.abs(np.diff(ys))) or 1.0
    max_resid_frac = float(np.max(np.abs(resid)) / spacing) if spacing else 1.0
    confidence = max(0.0, 1.0 - max_resid_frac)
    needs_review = (not monotonic) or max_resid_frac > max_outlier_frac or dropped

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "confidence": confidence,
        "needs_review": bool(needs_review),
        "monotonic": bool(monotonic),
        "n_ticks_used": len(xs),
        "dropped_outlier": dropped,
        "n_dropped": n_dropped,
    }


def value_at(calib, pixel_pos):
    return calib["slope"] * pixel_pos + calib["intercept"]
