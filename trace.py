#!/usr/bin/env python3
"""CLI for tracing curves out of graph images and overlaying normalized traces.

Usage:
    python trace.py extract GraphEx1.png --config config.csv --outdir output
    python trace.py overlay output/*.csv --out output/overlay.png --alpha 0.6

For a single graph image, give `config.csv` one row for that filename. For an
image containing several graphs stacked vertically (sharing one x-axis at the
bottom), give it one row per panel, numbered top to bottom via a `panel`
column (1, 2, 3, ...).
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from tracer import border, colors, extract, calibrate, config as cfgmod
from tracer.overlay import render_overlay
from tracer import pdf_scan
from tracer.output_store import OutputStore

BORDER_MARGIN = 4  # px to strip past the detected border line itself

# matplotlib doesn't know "yellow-green" by that exact name
_MPL_COLOR_FIX = {"yellow-green": "yellowgreen"}


def _unique_name(base, used):
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def cmd_extract(args):
    os.makedirs(args.outdir, exist_ok=True)
    basename = os.path.basename(args.image)

    cfg = cfgmod.load_config(args.config)
    if basename not in cfg:
        raise SystemExit(f"No config row(s) found for '{basename}' in {args.config}")
    panel_rows = cfg[basename]  # {panel_number: row_dict}
    num_panels = len(panel_rows)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    panel_boxes = border.detect_stacked_panels(gray, expected_count=num_panels)
    if len(panel_boxes) != num_panels:
        print(
            f"Warning: config lists {num_panels} panel(s) for '{basename}' but "
            f"{len(panel_boxes)} were detected in the image. Processing "
            f"min(the two)."
        )
    n = min(num_panels, len(panel_boxes))

    stem = os.path.splitext(basename)[0]
    debug_img = img.copy()
    m = BORDER_MARGIN
    used_names = set()

    for panel_num in range(1, n + 1):
        left, right, top, bottom = panel_boxes[panel_num - 1]
        row = panel_rows[panel_num]
        print(f"Panel {panel_num}: border left={left} right={right} top={top} bottom={bottom}")

        interior = img[top + m:bottom - m, left + m:right - m]
        line_masks = colors.find_line_masks(interior)
        print(f"  Detected {len(line_masks)} line color(s): {[l['name'] for l in line_masks]}")

        for line in line_masks:
            paths = extract.mask_to_paths(line["mask"])
            if not paths:
                print(f"  '{line['name']}': no continuous path found, skipping")
                continue

            for path in paths:
                x_px = path["x"] + left + m
                y_px = path["y"] + top + m

                x_data, y_data = calibrate.pixel_to_data(
                    x_px, y_px, (left, right, top, bottom),
                    row["x_min"], row["x_max"], row["y_min"], row["y_max"],
                )
                x_norm, y_norm = calibrate.normalize(
                    x_data, y_data, row["x_min"], row["x_max"], row["y_min"], row["y_max"]
                )

                out_df = pd.DataFrame({
                    "x_px": x_px, "y_px": y_px,
                    "x_data": x_data, "y_data": y_data,
                    "x_norm": x_norm, "y_norm": y_norm,
                    "color_name": line["name"],
                    "direction": path["direction"] or "",
                    "panel": panel_num,
                })

                base = f"{stem}_panel{panel_num}_{line['name']}" if num_panels > 1 else f"{stem}_{line['name']}"
                if path["direction"]:
                    base = f"{base}_{path['direction']}"
                name = _unique_name(base, used_names)
                out_path = os.path.join(args.outdir, f"{name}.csv")
                out_df.to_csv(out_path, index=False)
                label = path["direction"] or line["name"]
                print(f"  '{line['name']}' ({label}): {len(out_df)} points -> {out_path}")

                pts = np.stack([x_px, y_px], axis=1).astype(np.int32)
                cv2.polylines(debug_img, [pts], isClosed=False, color=(0, 0, 255), thickness=1)

    debug_path = os.path.join(args.outdir, f"{stem}_debug.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug overlay (extracted paths drawn in red on original): {debug_path}")


def cmd_overlay(args):
    traces = []
    for spec in args.csvs:
        path, _, label_override = spec.partition(":")
        df = pd.read_csv(path)
        stem = os.path.splitext(os.path.basename(path))[0]

        if "color_name" in df.columns and pd.notna(df["color_name"].iloc[0]):
            color_name = df["color_name"].iloc[0]
        else:
            color_name = stem.split("_")[-1]
        color = _MPL_COLOR_FIX.get(color_name, color_name)
        label = label_override or stem

        traces.append({
            "x_norm": df["x_norm"], "y_norm": df["y_norm"],
            "color": color, "label": label, "alpha": args.alpha,
        })

    render_overlay(
        traces, args.out,
        x_axis_label=args.xlabel, y_axis_label=args.ylabel,
    )
    print(f"Overlay saved to {args.out}")


def cmd_pdf_scan(args):
    store = OutputStore(args.outdir)
    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else None

    pdf_paths = []
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            if fname.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(args.input, fname))
    else:
        pdf_paths.append(args.input)

    for pdf_path in pdf_paths:
        print(f"Scanning {pdf_path} ...")
        try:
            pdf_scan.process_pdf(pdf_path, store, keywords=keywords)
        except Exception as e:
            print(f"  ERROR processing {pdf_path}: {e}")

    print(f"Done. Output in {args.outdir}/ (metadata.csv, traces.csv, review_report.csv)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Trace lines out of one graph image")
    p_extract.add_argument("image")
    p_extract.add_argument("--config", required=True, help="CSV with per-image/per-panel axis calibration")
    p_extract.add_argument("--outdir", default="output")
    p_extract.set_defaults(func=cmd_extract)

    p_overlay = sub.add_parser("overlay", help="Composite normalized traces into one image")
    p_overlay.add_argument("csvs", nargs="+", help="Trace CSVs (optionally 'path:label')")
    p_overlay.add_argument("--out", default="overlay.png")
    p_overlay.add_argument("--alpha", type=float, default=0.7)
    p_overlay.add_argument("--xlabel", default="x (normalized)")
    p_overlay.add_argument("--ylabel", default="y (normalized)")
    p_overlay.set_defaults(func=cmd_overlay)

    p_scan = sub.add_parser("pdf-scan", help="Stage 1: find figures in PDFs, trace + auto-label + auto-calibrate")
    p_scan.add_argument("input", help="A single PDF file or a folder of PDFs")
    p_scan.add_argument("--outdir", default="raw")
    p_scan.add_argument("--keywords", default=None,
                        help="Comma-separated caption keywords for relevance filtering (default: built-in list)")
    p_scan.set_defaults(func=cmd_pdf_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
