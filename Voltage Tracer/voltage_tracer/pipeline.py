"""End-to-end pipeline for one figure: contract + image -> data + report.

    contract -> preprocess (+metrics) -> skeletonize -> track -> smooth ->
    calibrate (+resample) -> validate -> write outputs

Deterministic and token-free: every step is Python. Writes one CSV per curve
(raw + smoothed + resampled columns) and a per-figure JSON summary of the
validation warnings, and optionally a raw+smoothed debug overlay.
"""
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voltage_tracer.calibrate import calibrate_all
from voltage_tracer.contract import Contract, load_contract
from voltage_tracer.metrics import FigureMetrics, compute_metrics
from voltage_tracer.preprocess import (
    color_mask,
    crop_to_plot_area,
    load_image,
    mask_gridlines_and_axes,
    mask_legend_region,
    mask_outside_data_region,
    mask_regions,
    mask_text_ink,
    remove_small_components,
)
from voltage_tracer.skeletonize import extract_column_runs, skeletonize_mask
from voltage_tracer.smooth import smooth_all
from voltage_tracer.tracker import (
    Track,
    bridge_occluded_segments,
    extend_terminal_drops,
    track_columns,
)
from voltage_tracer.validate import Finding, validate_all
from voltage_tracer.visualize import save_debug_overlay


@dataclass
class PipelineResult:
    figure: str
    tracks: list[Track]
    findings: list[Finding]
    csv_paths: list[Path]
    summary_path: Path
    overlay_path: Path | None


def _exclusion_mask(crop, contract: Contract, metrics: FigureMetrics):
    """Everything a curve must NOT be traced from: gridlines/axes, the
    auto-detected text/legend regions, an optional contract legend override,
    and the optional data-region restriction (all curve-free by assumption)."""
    x0 = int(round(contract.plot_bbox_px[0]))
    y0 = int(round(contract.plot_bbox_px[1]))
    legend_bbox = (
        None if contract.legend_bbox_px is None
        else (contract.legend_bbox_px[0] - x0, contract.legend_bbox_px[1] - y0,
              contract.legend_bbox_px[2] - x0, contract.legend_bbox_px[3] - y0)
    )
    return (
        mask_gridlines_and_axes(crop)
        # label boxes: mask only the achromatic ink, so a colored curve
        # passing through a title/label survives
        # tight glyph bounds (minimal dilation): masking only the glyph
        # strokes leaves an ACHROMATIC (black) curve passing through a text
        # annotation visible between glyphs, so the tracker bridges the small
        # per-glyph gaps and continues, rather than losing the whole curve
        | mask_text_ink(crop, metrics.text_regions,
                        chroma_max=metrics.d.text_chroma_max,
                        dark_lfrac=metrics.d.text_dark_lfrac,
                        dilate_px=1)
        # legend swatches: genuinely spurious colored marks, mask in full
        | mask_regions(crop.shape[:2], metrics.swatch_regions)
        | mask_legend_region(crop, override_bbox=legend_bbox)
        | mask_outside_data_region(crop.shape[:2], contract.data_region_frac)
    )


def trace_figure(
    crop: np.ndarray, contract: Contract, metrics: FigureMetrics
) -> list[Track]:
    """Preprocess -> per-color skeletonize/track -> terminal-drop extension.
    Returns tracks in crop-local pixel space (not yet smoothed/calibrated)."""
    exclude = _exclusion_mask(crop, contract, metrics)
    groups: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for s in contract.series:
        groups[s.color_rgb].append(s.id)

    # Strict per-colour masks up front: the bridging pass needs each group's
    # OTHER colours to tell an occlusion gap (curve ran under another colour)
    # from a real end.
    group_cmask = {
        rgb: remove_small_components(
            color_mask(crop, rgb, tolerance=metrics.color_tol) & ~exclude,
            min_area=metrics.speckle_min_area,
        )
        for rgb in groups
    }

    tracks: list[Track] = []
    for rgb, ids in groups.items():
        cmask = group_cmask[rgb]
        runs = extract_column_runs(skeletonize_mask(cmask), merge_gap=metrics.merge_gap)
        columns = {x: [mean for _, _, mean in rs] for x, rs in runs.items()}
        group_tracks = track_columns(columns, contract, metrics, series_ids=ids)
        # reconnect a curve hidden under another colour for a long stretch,
        # BEFORE the terminal-drop pass so the drop extends from the recovered
        # (later) segment's end. Occlusion evidence = the other colours' runs.
        other = np.zeros(cmask.shape, dtype=bool)
        for orgb, om in group_cmask.items():
            if orgb != rgb:
                other |= om
        other_runs = extract_column_runs(skeletonize_mask(other), merge_gap=metrics.merge_gap)
        bridge_occluded_segments(group_tracks, columns, other_runs, contract, metrics)
        # a locally-lenient mask feeds only the terminal-drop extension: a
        # steep fall dilutes its ink below the strict tolerance
        lenient = color_mask(crop, rgb, tolerance=metrics.lenient_tol) & ~exclude
        ext_runs = extract_column_runs(
            skeletonize_mask(lenient), merge_gap=metrics.merge_gap
        )
        extend_terminal_drops(group_tracks, ext_runs, metrics)
        tracks += group_tracks
    return tracks


def _write_curve_csv(track: Track, path: Path) -> None:
    """One CSV per curve: raw and smoothed share the capacity grid (smoothing
    changes only voltage); the resampled smoothed curve is on its own uniform
    capacity grid of the same length.

    `bridged` marks each raw point that was interpolated across an occlusion
    (a reconnected hidden span, not a seen pixel -- `info_raw.observed` False),
    so a reviewer can drop or down-weight reconstructed stretches in post."""
    raw = track.data_raw or []
    smooth = track.data_smooth or raw
    resamp = track.data_resampled or []
    info = track.info_raw or []
    n = max(len(raw), len(resamp))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "capacity", "voltage_raw", "voltage_smooth",
            "capacity_uniform", "voltage_resampled", "bridged",
        ])
        for i in range(n):
            cap = raw[i][0] if i < len(raw) else ""
            vraw = raw[i][1] if i < len(raw) else ""
            vsm = smooth[i][1] if i < len(smooth) else ""
            cu = resamp[i][0] if i < len(resamp) else ""
            vr = resamp[i][1] if i < len(resamp) else ""
            bridged = (0 if info[i].observed else 1) if i < len(info) else ""
            w.writerow([cap, vraw, vsm, cu, vr, bridged])


def _summary(figure: str, metrics: FigureMetrics, tracks: list[Track],
             findings: list[Finding]) -> dict:
    def rng(data, idx):
        vals = [p[idx] for p in (data or [])]
        return [min(vals), max(vals)] if vals else None
    return {
        "figure": figure,
        "status": "success",
        "n_curves": len(tracks),
        "metrics": {
            "line_width_px": metrics.line_width_px,
            "plot_w_px": metrics.plot_w_px,
            "plot_h_px": metrics.plot_h_px,
            "n_text_regions": len(metrics.text_regions),
        },
        "curves": [
            {
                "id": t.series_id,
                "direction": t.direction,
                "n_points": len(t.points_raw),
                "capacity_range": rng(t.data_raw, 0),
                "voltage_range": rng(t.data_raw, 1),
                # a reconnected occlusion span makes the curve lower-confidence
                "bridged": any(not i.observed for i in t.info_raw),
                "bridged_points": sum(1 for i in t.info_raw if not i.observed),
            }
            for t in tracks
        ],
        "n_findings": len(findings),
        "findings": [
            {
                "severity": f.severity, "check": f.check, "curve_id": f.curve_id,
                "description": f.description, "x_range": list(f.x_range) if f.x_range else None,
            }
            for f in findings
        ],
    }


def run_pipeline(
    image_path: str, contract_path: str, output_dir: str, debug: bool = False
) -> PipelineResult:
    """Full deterministic run for one figure. Writes per-curve CSVs + a JSON
    summary (and a debug overlay if `debug`), and returns a `PipelineResult`."""
    image = load_image(image_path)
    contract = load_contract(contract_path)
    stem = Path(image_path).stem
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    crop = crop_to_plot_area(image, contract.plot_bbox_px)
    metrics = compute_metrics(crop, contract)

    tracks = trace_figure(crop, contract, metrics)
    smooth_all(tracks, metrics, contract)
    calibrate_all(tracks, contract)
    findings = validate_all(tracks, contract, metrics)

    csv_paths = []
    for t in tracks:
        p = outdir / f"{stem}__{t.series_id}.csv"
        _write_curve_csv(t, p)
        csv_paths.append(p)

    summary_path = outdir / f"{stem}__summary.json"
    summary_path.write_text(json.dumps(_summary(stem, metrics, tracks, findings), indent=2))

    overlay_path = None
    if debug:
        overlay_path = outdir / f"{stem}__overlay.png"
        save_debug_overlay(image, contract, metrics, tracks, overlay_path)

    return PipelineResult(stem, tracks, findings, csv_paths, summary_path, overlay_path)
