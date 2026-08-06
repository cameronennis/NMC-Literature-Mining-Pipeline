"""OCR a single bold panel-letter tag (e.g. "a", "b", "c") printed near a
sub-panel's corner, used to pick the matching caption clause
(pdf_ingest.split_caption_by_panel) for that specific panel."""
import cv2
import pytesseract


def read_panel_letter(gray_crop):
    """Returns a single lowercase letter, or None if nothing legible was
    found (caller falls back to the whole caption text at lower
    confidence)."""
    if gray_crop.size == 0:
        return None
    text = pytesseract.image_to_string(
        gray_crop, config="--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz"
    ).strip().lower()
    return text if len(text) == 1 and text.isalpha() else None


def find_panel_letter(gray, box, search=36):
    """Try both top corners just outside/inside the given (left, right, top,
    bottom) box -- paners print the tag in whichever corner has clear
    space, most often top-left."""
    left, right, top, bottom = box
    h, w = gray.shape
    candidates = [
        gray[max(0, top - 6):top + search, max(0, left - 6):left + search],
        gray[max(0, top - 6):top + search, max(0, right - search):min(w, right + 6)],
    ]
    for crop in candidates:
        letter = read_panel_letter(crop)
        if letter:
            return letter
    return None
