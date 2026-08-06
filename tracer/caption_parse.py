"""Parse a figure caption (real PDF text, not OCR) for relevance and for a
curve-color -> value mapping, used whenever a figure has no in-plot legend
at all (validated against a real caption: "...0.191 mA cm-2 (red profiles)
and 1.5 mA cm-2 (blue profiles)")."""
import re

DEFAULT_KEYWORDS = [
    "c-rate", "c rate", "cycling", "charge/discharge", "charge-discharge",
    "specific capacity", "galvanostatic", "current density",
    "areal capacity", "voltage profile", "rate performance",
]

# Shared with axis_ocr.UNIT_VOCAB in spirit, but written as a regex
# alternation matching how units actually appear in running prose (longest
# first, so "mA h cm-2" matches before the "mA cm-2" prefix it contains).
_UNIT_ALT = (
    r"mA\s*h\s*cm-?\d|mAh\s*cm-?\d|mA\s*cm-?\d|"
    r"mAh\s*g-?\d|mA\s*h\s*g-?\d|mg\s*cm-?\d|W\s*/?\s*cm2|mW\s*/?\s*cm2|"
    r"°?C\b|%|V\b"
)
_VALUE_UNIT_RE = re.compile(rf"(-?\d+\.?\d*)\s*({_UNIT_ALT})", re.IGNORECASE)
_COLOR_PAREN_RE = re.compile(
    rf"(-?\d+\.?\d*)\s*({_UNIT_ALT})\s*\(\s*([a-z]+)\s*(?:profiles?|curves?|lines?)\s*\)",
    re.IGNORECASE,
)
_COMMA_LIST_RE = re.compile(
    rf"((?:-?\d+\.?\d*\s*(?:,|and)\s*)+-?\d+\.?\d*)\s*({_UNIT_ALT})",
    re.IGNORECASE,
)

_COLOR_WORDS = {
    "red", "blue", "green", "black", "orange", "purple", "magenta", "cyan",
    "yellow", "navy", "olive", "gray", "grey", "pink", "brown", "violet",
}


def _normalize(text):
    return text.replace("−", "-").replace("⁻", "-")


def is_relevant(caption_text, keywords=None):
    """Keyword-allowlist gate: is this figure/panel worth extracting at
    all?"""
    keywords = keywords or DEFAULT_KEYWORDS
    text_lower = caption_text.lower()
    return any(kw in text_lower for kw in keywords)


def find_value_unit_tokens(caption_text):
    """Every bare (value, unit) token in the text -- used by
    metadata_extract.py for current density/C-rate/loading/etc, not just
    color matching."""
    text = _normalize(caption_text)
    return [(float(m.group(1)), m.group(2)) for m in _VALUE_UNIT_RE.finditer(text)]


def find_color_value_pairs(caption_text):
    """Explicit "<value><unit> (<color> profiles)" mentions -- the most
    direct, highest-trust way to map a curve color to its value when there's
    no in-plot legend. Returns (value, unit, color_word) tuples."""
    text = _normalize(caption_text)
    return [
        (float(m.group(1)), m.group(2), m.group(3).lower())
        for m in _COLOR_PAREN_RE.finditer(text)
    ]


def find_comma_list_values(caption_text):
    """"at rates of 0.1, 0.2, 0.5, 1 and 2 C" style lists: a run of bare
    numbers sharing one trailing unit. Returns (values, unit) or None."""
    text = _normalize(caption_text)
    m = _COMMA_LIST_RE.search(text)
    if not m:
        return None
    numbers_text, unit = m.group(1), m.group(2)
    values = [float(v) for v in re.split(r"\s*,\s*|\s+and\s+", numbers_text.strip()) if v]
    return values, unit


def _names_match(color_word, detected_name):
    a, b = color_word.lower(), detected_name.lower()
    return a == b or a in b or b in a


def match_values_to_colors(caption_text, detected_color_names):
    """Map each distinct detected curve *color* (not path -- callers must
    already have collapsed charge/discharge duplicates of the same color to
    one entry) to a value parsed from the caption. Returns a dict
    {color_name: {"value": float, "unit": str, "source": str,
    "needs_review": bool}}. Never fabricates a mapping with zero textual
    support."""
    results = {}

    # Tier 1: explicit "<value><unit> (<color> profiles)" -- highest trust.
    for value, unit, color_word in find_color_value_pairs(caption_text):
        for name in detected_color_names:
            if _names_match(color_word, name) and name not in results:
                results[name] = {
                    "value": value, "unit": unit,
                    "source": "caption_color_word", "needs_review": False,
                }

    remaining = [n for n in detected_color_names if n not in results]
    if not remaining:
        return results

    # Tier 2: a bare comma/and-list ("0.1, 0.2, 0.5 C") with no color word --
    # only usable if the count matches exactly, and always flagged even
    # though it's auto-applied, since it's an order-based guess.
    comma = find_comma_list_values(caption_text)
    if comma is not None:
        values, unit = comma
        if len(values) == len(remaining):
            for name, value in zip(remaining, values):
                results[name] = {
                    "value": value, "unit": unit,
                    "source": "order_guess", "needs_review": True,
                }
    return results
