"""Append-only consolidated output for the PDF batch pipeline: a handful of
growing CSVs, never one file per curve (at ~1000 papers x several curves
each that would be thousands of files to manage)."""
import csv
import os

METADATA_COLUMNS = [
    "curve_id", "pdf", "page", "figure_index", "panel", "panel_letter",
    "color_name", "direction", "label_value", "label_unit", "label_source",
    "label_needs_review",
    "x_min", "x_max", "x_unit", "y_min", "y_max", "y_unit",
    "axis_confidence", "axis_needs_review",
    "cathode_composition", "areal_loading_mg_cm2", "stack_pressure_MPa",
    "temperature_C", "electrolyte_type",
    "current_density_mA_cm2", "current_density_source",
    "needs_review", "review_reason",
]
TRACES_COLUMNS = ["curve_id", "x_px", "y_px", "x_data", "y_data", "x_norm", "y_norm"]
REVIEW_COLUMNS = ["curve_id", "field", "reason", "debug_image"]


def _append_rows(path, columns, rows):
    if not rows:
        return
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


class OutputStore:
    """Wraps the three append-only CSVs for one `raw/` output directory."""

    def __init__(self, outdir):
        self.outdir = outdir
        self.debug_dir = os.path.join(outdir, "debug")
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.metadata_path = os.path.join(outdir, "metadata.csv")
        self.traces_path = os.path.join(outdir, "traces.csv")
        self.review_path = os.path.join(outdir, "review_report.csv")

    def add_curve(self, metadata_row, trace_points):
        """metadata_row: dict matching (a subset of) METADATA_COLUMNS.
        trace_points: list of dicts matching (a subset of) TRACES_COLUMNS,
        each missing `curve_id` (filled in here)."""
        curve_id = metadata_row["curve_id"]
        _append_rows(self.metadata_path, METADATA_COLUMNS, [metadata_row])
        for p in trace_points:
            p.setdefault("curve_id", curve_id)
        _append_rows(self.traces_path, TRACES_COLUMNS, trace_points)
        if metadata_row.get("needs_review"):
            _append_rows(self.review_path, REVIEW_COLUMNS, [{
                "curve_id": curve_id,
                "field": "curve",
                "reason": metadata_row.get("review_reason", ""),
                "debug_image": metadata_row.get("_debug_image", ""),
            }])


def load_manual_overrides(path):
    """Same shape as METADATA_COLUMNS, keyed by curve_id. Checked before
    trusting any OCR'd/parsed value for that curve, so a flagged row can be
    hand-corrected and only that row re-processed."""
    if not path or not os.path.exists(path):
        return {}
    overrides = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("curve_id"):
                overrides[row["curve_id"]] = row
    return overrides
