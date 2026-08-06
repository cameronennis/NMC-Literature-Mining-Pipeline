"""Locate figures and their captions inside a PDF page.

Validated against a real journal PDF: multi-panel figures are typically one
single flattened raster image (an embedded XObject), not live vector paths,
so figure localization is a direct PyMuPDF lookup rather than a page-wide CV
search. Captions are found geometrically (the text block sitting just below
the figure's bbox that starts with "Figure N.") rather than by searching the
raw text stream, since content-stream order doesn't match visual layout.
"""
import re

import cv2
import numpy as np


def render_page(doc, page_index, dpi=200):
    """Full page as a BGR numpy array, for the rare vector-figure fallback."""
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def find_figure_images(page, min_size=150):
    """Return a list of {"bbox": fitz.Rect (PDF points), "xref": int,
    "width": int, "height": int} for each embedded raster image on the page,
    largest first. `min_size` filters out small decorative images (logos,
    icons) in native pixel dimensions."""
    results = []
    for info in page.get_image_info(xrefs=True):
        if info.get("width", 0) < min_size or info.get("height", 0) < min_size:
            continue
        results.append({
            "bbox": info["bbox"],
            "xref": info["xref"],
            "width": info["width"],
            "height": info["height"],
        })
    results.sort(key=lambda r: r["width"] * r["height"], reverse=True)
    return results


def get_native_image(doc, xref):
    """Decode an embedded image at its native resolution (better for OCR
    than re-rasterizing the page at an arbitrary DPI)."""
    data = doc.extract_image(xref)["image"]
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def page_has_text_layer(page, min_chars=200):
    """False for a scanned/image-only page, gating the OCR fallback for
    text extraction."""
    return len(page.get_text()) >= min_chars


def find_caption_block(page, image_bbox):
    """Find the caption for a figure: the text block sitting at/below the
    image's bottom edge, sorted top-to-bottom, whose text starts with
    "Figure " -- validated more reliable than searching the raw text stream,
    which can return unrelated sentences that merely mention "Figure N"."""
    blocks = page.get_text("blocks")
    below = sorted(
        (b for b in blocks if b[1] >= image_bbox[3] - 5),
        key=lambda b: b[1],
    )
    for b in below:
        text = b[4].strip()
        if re.match(r"^Figure\s+\S", text):
            return text
    return None


def split_caption_by_panel(caption_text):
    """Split a caption into {letter: clause} using "(a)", "(b)", ... markers.
    Figures without lettered sub-panels return {} (caller should fall back
    to the whole caption text)."""
    matches = list(re.finditer(r"\(([a-z])\)", caption_text))
    if not matches:
        return {}
    clauses = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(caption_text)
        clauses[m.group(1)] = caption_text[start:end].strip()
    return clauses
