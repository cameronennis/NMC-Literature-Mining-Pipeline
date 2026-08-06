"""Extract the parameters the research goal actually groups by (cathode
composition, areal loading, stack pressure, temperature, electrolyte type),
and resolve each curve's areal current density -- the one hard gate for
keeping a curve at all.

Two text sources feed every field: the caption/body (`caption_text`, real
PDF text, high trust) and whatever plain text is OCR'd out of the figure
itself (`figure_text` -- some papers print things like "S-NCM88: ~5.1 mg
cm-2" or "Ambient temperature: 30 C" directly on the plot as a rasterized
annotation, not in the caption). Caption text wins on conflict, per the
"assume in-text values are most accurate" default.
"""
import re

import pytesseract

_MPA_RE = re.compile(r"(-?\d+\.?\d*)\s*MPa", re.IGNORECASE)
_TEMP_RE = re.compile(r"(-?\d+\.?\d*)\s*°\s*C")
_LOADING_RE = re.compile(r"(-?\d+\.?\d*)\s*mg\s*cm\S{0,2}", re.IGNORECASE)
_CATHODE_RE = re.compile(
    r"\bS?-?N[CM]M?\d{2,3}\b|\bNMC\s?\d{3}\b|\bNCM\s?\d{3}\b|"
    r"LiNi[\d.]+Co[\d.]+Mn[\d.]+O2",
    re.IGNORECASE,
)


def ocr_full_text(interior_bgr):
    """Grab every bit of plain text OCR can find anywhere in a figure crop
    -- used for in-plot annotations (mass loading, temperature) that live
    outside the caption/legend structures the other modules already parse."""
    import cv2
    gray = cv2.cvtColor(interior_bgr, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config="--psm 11")


def _first_match(pattern, *texts):
    """Search caption text first (higher trust), then figure-OCR text."""
    for text in texts:
        if not text:
            continue
        m = pattern.search(text)
        if m:
            return m, text
    return None, None


def extract_fields(caption_text=None, figure_text=None):
    """Returns {field: {"value":..., "source": "caption"/"figure_ocr",
    "needs_review": bool}} for whichever of the 5 fields were found. Missing
    fields are simply absent from the dict -- every field is independently
    nullable."""
    fields = {}

    m, src = _first_match(_MPA_RE, caption_text, figure_text)
    if m:
        fields["stack_pressure_MPa"] = {
            "value": float(m.group(1)),
            "source": "caption" if src == caption_text else "figure_ocr",
            "needs_review": False,
        }

    m, src = _first_match(_TEMP_RE, caption_text, figure_text)
    if m:
        fields["temperature_C"] = {
            "value": float(m.group(1)),
            "source": "caption" if src == caption_text else "figure_ocr",
            "needs_review": False,
        }

    m, src = _first_match(_LOADING_RE, caption_text, figure_text)
    if m:
        fields["areal_loading_mg_cm2"] = {
            "value": float(m.group(1)),
            "source": "caption" if src == caption_text else "figure_ocr",
            "needs_review": False,
        }

    m, src = _first_match(_CATHODE_RE, caption_text, figure_text)
    if m:
        fields["cathode_composition"] = {
            "value": m.group(0),
            "source": "caption" if src == caption_text else "figure_ocr",
            "needs_review": False,
        }

    # Electrolyte type is highly paper-specific shorthand (e.g. "A-LTC"),
    # not attempted via a generic regex here -- left to a manual_overrides
    # entry or a future paper-specific hint; recorded as absent rather than
    # guessed.

    # Cross-check: if both sources produced a value for the same field and
    # they disagree, keep the caption's (higher trust) but flag it.
    for field_name, pattern in (
        ("stack_pressure_MPa", _MPA_RE), ("temperature_C", _TEMP_RE),
        ("areal_loading_mg_cm2", _LOADING_RE),
    ):
        if field_name not in fields or not caption_text or not figure_text:
            continue
        m_cap = pattern.search(caption_text)
        m_fig = pattern.search(figure_text)
        if m_cap and m_fig and float(m_cap.group(1)) != float(m_fig.group(1)):
            fields[field_name]["needs_review"] = True

    return fields


def resolve_current_density(value, unit, areal_loading_mg_cm2=None,
                             reference_capacity_mAh_g=None,
                             reference_capacity_mAh_cm2=None):
    """Resolve a curve's areal current density (mA/cm2), in priority order:
    direct value > derived from C-rate x reference capacity > unresolved.
    Returns (current_density_mA_cm2 or None, source_str)."""
    unit_norm = unit.lower().replace(" ", "").replace("-", "")

    if "macm2" in unit_norm or "mah cm2" in unit.lower():
        return value, "direct"

    if unit_norm in ("c", "crate"):
        if reference_capacity_mAh_cm2 is not None:
            return value * reference_capacity_mAh_cm2, "derived_explicit_areal_capacity"
        if reference_capacity_mAh_g is not None and areal_loading_mg_cm2 is not None:
            areal_capacity = reference_capacity_mAh_g * areal_loading_mg_cm2 / 1000.0
            return value * areal_capacity, "derived_loading_and_capacity"
        return None, "unresolved_missing_reference_capacity"

    return None, f"unsupported_unit:{unit}"


def lowest_rate_reference_capacity(rate_value_pairs):
    """Given [(rate_or_current, capacity_value), ...] for every curve in a
    figure, approximate the theoretical/reference capacity from the
    lowest-rate curve's own capacity, per the guidance that a low-enough
    rate is "somewhat accurate" as a stand-in when the paper never states a
    theoretical capacity explicitly. Returns (capacity, is_approximation)."""
    if not rate_value_pairs:
        return None, True
    lowest = min(rate_value_pairs, key=lambda p: p[0])
    return lowest[1], True
