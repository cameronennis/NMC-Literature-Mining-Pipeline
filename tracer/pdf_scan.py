"""Stage 1 orchestration: scan a batch of PDFs, find voltage-profile
figures, trace their curves, and label/calibrate everything automatically.
This is the expensive, OCR-heavy pass -- see compile.py for the cheap,
PDF-free Stage 2 that turns its output into power vs. time."""
import os
import re

import cv2
import fitz
import numpy as np

from . import (axis_ocr, border, calibrate, caption_parse, colors, extract,
               legend_ocr, metadata_extract, panel_letter, pdf_ingest)

BORDER_MARGIN = 4


def _curve_id(pdf_stem, page, fig_idx, letter, panel_num, box, color, direction):
    # Letter OCR fails often enough (small bold single-char tags at native
    # figure resolution) that two *different* boxes can both fall back to
    # the same "x1" placeholder tag; anchor the tag to the box's own
    # position too so that case doesn't collide, panel-letter-detection
    # accuracy aside.
    tag = f"{letter}{panel_num}" if letter else f"x{box[0]}_{box[2]}_{panel_num}"
    d = direction or "na"
    return f"{pdf_stem}_p{page}_fig{fig_idx}_{tag}_{color}_{d}"


def _unique_id(base, used_ids):
    """Two physically distinct curves can collide on the same base id --
    e.g. a merged-hue color bucket (colors.py can't always tell "blue" and
    "navy" apart) produces two "charge" tracks under one color name. Append
    a counter rather than silently overwriting one with the other."""
    if base not in used_ids:
        used_ids.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used_ids:
        i += 1
    unique = f"{base}_{i}"
    used_ids.add(unique)
    return unique


def _group_by_letter_column(boxes_with_letters, tol=15):
    """Boxes sharing a letter and roughly the same left/right column are one
    vertically-stacked group (e.g. Figure 4's 5-panel "c" stack); assign
    each a 1-indexed panel number top to bottom. A lone box (no stack-mates)
    just gets panel_num=1."""
    groups = {}
    for box, letter in boxes_with_letters:
        left, right = box[0], box[1]
        key = None
        for (l, r), letter2 in list(groups.keys()):
            if letter2 == letter and abs(l - left) <= tol and abs(r - right) <= tol:
                key = ((l, r), letter2)
                break
        if key is None:
            key = ((left, right), letter)
            groups[key] = []
        groups[key].append(box)

    result = []
    for (_, letter), boxes in groups.items():
        boxes = sorted(boxes, key=lambda b: b[2])
        for i, box in enumerate(boxes, start=1):
            result.append((box, letter, i))
    return result


def _calibrate_edge(gray, edge, box, inherit_from=None):
    pairs = axis_ocr.read_axis_labels(gray, edge, box)
    calib = axis_ocr.calibrate_axis(pairs)
    source = "ocr"
    if (calib is None or calib["needs_review"]) and inherit_from is not None:
        inherited = inherit_from.get(edge)
        if inherited is not None:
            calib, source = inherited, "inherited"
    return calib, source


def _distinct_colors(line_masks):
    seen = {}
    for lm in line_masks:
        seen.setdefault(lm["name"], lm["color_bgr"])
    return seen


def _resolve_labels(interior, rejected, line_masks, caption_clause):
    """One label per distinct color (never per charge/discharge path --
    those get the same label broadcast to both)."""
    distinct = _distinct_colors(line_masks)
    labels = {}

    legend_hits = legend_ocr.find_legend_labels(interior, rejected, line_masks)
    for hit in legend_hits:
        name = hit["matched_curve"]
        if name in distinct and name not in labels:
            m = re.match(r"(-?\d+\.?\d*)", hit["text"])
            if not m:
                continue
            labels[name] = {
                "value": float(m.group(1)), "unit": None,
                "source": "legend_" + (hit["color_source"] or "unknown"),
                "needs_review": hit["needs_review"],
            }

    remaining = [n for n in distinct if n not in labels]
    if remaining and caption_clause:
        caption_matches = caption_parse.match_values_to_colors(caption_clause, remaining)
        labels.update(caption_matches)

    return labels


def process_figure(gray, native_bgr, box, letter, panel_num, page_index, fig_idx,
                    pdf_stem, caption_clause, store, used_ids, x_inherit=None, y_inherit=None):
    left, right, top, bottom = box
    m = BORDER_MARGIN
    interior = native_bgr[top + m:bottom - m, left + m:right - m]
    if interior.size == 0:
        return

    line_masks, rejected = colors.find_line_masks(interior, return_rejected=True)
    if not line_masks:
        return

    x_calib, x_source = _calibrate_edge(gray, "bottom", box, x_inherit)
    y_calib, y_source = _calibrate_edge(gray, "left", box, y_inherit)
    if x_calib is None or y_calib is None:
        return

    x_min, x_max = axis_ocr.value_at(x_calib, left), axis_ocr.value_at(x_calib, right)
    y_min, y_max = axis_ocr.value_at(y_calib, bottom), axis_ocr.value_at(y_calib, top)
    x_unit = axis_ocr.read_axis_unit(gray, box, "bottom")
    y_unit = axis_ocr.read_axis_unit(gray, box, "left")

    labels = _resolve_labels(interior, rejected, line_masks, caption_clause)

    figure_text = metadata_extract.ocr_full_text(interior)
    fields = metadata_extract.extract_fields(caption_text=caption_clause, figure_text=figure_text)

    # Reference (theoretical/practical) capacity for C-rate -> current
    # density conversion, approximated from the panel's own lowest-rate
    # curve if nothing explicit was found in text.
    rate_capacity_pairs = []

    debug_img = native_bgr.copy()
    curves_this_panel = []

    for line in line_masks:
        paths = extract.mask_to_paths(line["mask"])
        for path in paths:
            x_px = path["x"] + left + m
            y_px = path["y"] + top + m
            x_data, y_data = calibrate.pixel_to_data(
                x_px, y_px, box, x_min, x_max, y_min, y_max)
            x_norm, y_norm = calibrate.normalize(x_data, y_data, x_min, x_max, y_min, y_max)

            label = labels.get(line["name"])
            if label and label["unit"] in (None, "C", "c-rate"):
                rate_capacity_pairs.append((label["value"], float(np.max(x_data))))

            curves_this_panel.append({
                "color": line["name"], "direction": path["direction"],
                "x_px": x_px, "y_px": y_px, "x_data": x_data, "y_data": y_data,
                "x_norm": x_norm, "y_norm": y_norm,
            })

            pts = np.stack([x_px, y_px], axis=1).astype(np.int32)
            cv2.polylines(debug_img, [pts], False, (0, 0, 255), 1)

    ref_capacity, ref_is_approx = (None, False)
    if rate_capacity_pairs:
        ref_capacity, ref_is_approx = metadata_extract.lowest_rate_reference_capacity(
            rate_capacity_pairs)

    for c in curves_this_panel:
        color, direction = c["color"], c["direction"]
        curve_id = _unique_id(
            _curve_id(pdf_stem, page_index + 1, fig_idx, letter, panel_num, box, color, direction),
            used_ids,
        )
        label = labels.get(color)

        current_density, cd_source = None, "no_label"
        if label is not None:
            current_density, cd_source = metadata_extract.resolve_current_density(
                label["value"], label["unit"] or "mA cm-2",
                areal_loading_mg_cm2=fields.get("areal_loading_mg_cm2", {}).get("value"),
                reference_capacity_mAh_g=ref_capacity,
            )

        needs_review = bool(
            x_calib["needs_review"] or y_calib["needs_review"]
            or (label is None) or (label and label["needs_review"])
            or current_density is None or ref_is_approx
        )
        reasons = []
        if label is None:
            reasons.append("no label resolved for this color")
        if current_density is None:
            reasons.append(f"current density unresolved ({cd_source})")
        if ref_is_approx:
            reasons.append("reference capacity approximated from lowest-rate curve")

        metadata_row = {
            "curve_id": curve_id, "pdf": pdf_stem, "page": page_index + 1,
            "figure_index": fig_idx, "panel": panel_num, "panel_letter": letter or "",
            "color_name": color, "direction": direction or "",
            "label_value": label["value"] if label else "",
            "label_unit": (label["unit"] if label else "") or "",
            "label_source": label["source"] if label else "",
            "label_needs_review": label["needs_review"] if label else True,
            "x_min": x_min, "x_max": x_max, "x_unit": x_unit or "",
            "y_min": y_min, "y_max": y_max, "y_unit": y_unit or "",
            "axis_confidence": min(x_calib["confidence"], y_calib["confidence"]),
            "axis_needs_review": x_calib["needs_review"] or y_calib["needs_review"],
            "cathode_composition": fields.get("cathode_composition", {}).get("value", ""),
            "areal_loading_mg_cm2": fields.get("areal_loading_mg_cm2", {}).get("value", ""),
            "stack_pressure_MPa": fields.get("stack_pressure_MPa", {}).get("value", ""),
            "temperature_C": fields.get("temperature_C", {}).get("value", ""),
            "electrolyte_type": fields.get("electrolyte_type", {}).get("value", ""),
            "current_density_mA_cm2": current_density if current_density is not None else "",
            "current_density_source": cd_source,
            "needs_review": needs_review,
            "review_reason": "; ".join(reasons),
        }
        trace_points = [
            {"x_px": xp, "y_px": yp, "x_data": xd, "y_data": yd, "x_norm": xn, "y_norm": yn}
            for xp, yp, xd, yd, xn, yn in zip(
                c["x_px"], c["y_px"], c["x_data"], c["y_data"], c["x_norm"], c["y_norm"])
        ]
        store.add_curve(metadata_row, trace_points)

    # Same box-position disambiguation as _curve_id: a missing panel letter
    # must not let two different boxes collide on the same debug filename.
    tag = f"{letter}{panel_num}" if letter else f"x{left}_{top}_{panel_num}"
    debug_path = os.path.join(
        store.debug_dir, f"{pdf_stem}_p{page_index+1}_fig{fig_idx}_{tag}.png")
    cv2.imwrite(debug_path, debug_img[top - 10:bottom + 10, left - 10:right + 10])
    return {"x": x_calib, "y": y_calib}, box


def process_pdf(pdf_path, store, keywords=None):
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    used_ids = set()  # shared across the whole PDF so no two boxes/panels can collide

    for page_index in range(doc.page_count):
        page = doc[page_index]
        for fig_idx, img_info in enumerate(pdf_ingest.find_figure_images(page)):
            native_bgr = pdf_ingest.get_native_image(doc, img_info["xref"])
            gray = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2GRAY)

            caption = pdf_ingest.find_caption_block(page, img_info["bbox"])
            if not caption or not caption_parse.is_relevant(caption, keywords):
                continue
            panel_clauses = pdf_ingest.split_caption_by_panel(caption)

            try:
                boxes = border.find_all_boxes(gray)
            except Exception:
                boxes = []
            if not boxes:
                continue

            boxes_with_letters = []
            for box in boxes:
                letter = panel_letter.find_panel_letter(gray, box)
                boxes_with_letters.append((box, letter))
            grouped = _group_by_letter_column(boxes_with_letters)

            # x-axis calibration inheritance for stacked groups: process
            # bottom-most panel of each column first so others can borrow it.
            grouped.sort(key=lambda g: -g[0][3])
            column_x_calib = {}

            for box, letter, panel_num in grouped:
                clause = panel_clauses.get(letter) if letter else caption
                if not clause or not caption_parse.is_relevant(clause, keywords):
                    continue
                col_key = (round(box[0] / 15), round(box[1] / 15))
                result = process_figure(
                    gray, native_bgr, box, letter, panel_num, page_index, fig_idx,
                    pdf_stem, clause, store, used_ids,
                    x_inherit=column_x_calib.get(col_key),
                )
                if result:
                    calibs, _ = result
                    column_x_calib.setdefault(col_key, {})["bottom"] = calibs["x"]
