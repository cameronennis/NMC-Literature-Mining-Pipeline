import csv
import json

import cv2
import numpy as np
import pytest

from voltage_tracer.pipeline import PipelineResult, run_pipeline


def _synthetic_figure(tmp_path):
    """A minimal real-ish figure: black axis box, one red discharge curve,
    one blue charge curve, on white. Returns (image_path, contract_path)."""
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 20), (360, 260), (0, 0, 0), 2)  # axis box
    xs = np.arange(45, 340)
    # discharge: falls (voltage down => pixel y up)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        y0 = int(40 + 0.6 * (x0 - 45)); y1 = int(40 + 0.6 * (x1 - 45))
        cv2.line(img, (x0, y0), (x1, y1), (220, 30, 30), 2, cv2.LINE_AA)
    # charge: rises (pixel y down)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        y0 = int(240 - 0.6 * (x0 - 45)); y1 = int(240 - 0.6 * (x1 - 45))
        cv2.line(img, (x0, y0), (x1, y1), (30, 30, 220), 2, cv2.LINE_AA)
    image_path = tmp_path / "fig.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    contract = {
        "plot_bbox_px": [40, 20, 360, 260],
        "x_axis": {"scale": "linear", "ticks_px": [40, 360], "ticks_val": [0, 320]},
        "y_axis": {"scale": "linear", "ticks_px": [260, 20], "ticks_val": [2.0, 4.0]},
        "series": [
            {"id": "red_dis", "color_rgb": [220, 30, 30], "label": "d",
             "direction": "decreasing"},
            {"id": "blue_chg", "color_rgb": [30, 30, 220], "label": "c",
             "direction": "increasing"},
        ],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return str(image_path), str(contract_path)


def test_run_pipeline_writes_csvs_and_summary(tmp_path):
    img, con = _synthetic_figure(tmp_path)
    out = tmp_path / "out"
    result = run_pipeline(img, con, str(out), debug=True)

    assert isinstance(result, PipelineResult)
    assert len(result.tracks) == 2
    assert len(result.csv_paths) == 2
    assert all(p.exists() for p in result.csv_paths)
    assert result.summary_path.exists()
    assert result.overlay_path is not None and result.overlay_path.exists()


def test_csv_has_raw_smooth_and_resampled_columns(tmp_path):
    img, con = _synthetic_figure(tmp_path)
    result = run_pipeline(img, con, str(tmp_path / "out"))
    with result.csv_paths[0].open() as f:
        rows = list(csv.DictReader(f))
    assert rows
    for col in ("capacity", "voltage_raw", "voltage_smooth",
                "capacity_uniform", "voltage_resampled"):
        assert col in rows[0]
    # values parse as floats and capacities are physical (0..320 mAh)
    caps = [float(r["capacity"]) for r in rows]
    assert 0 <= min(caps) and max(caps) <= 320


def test_summary_json_structure(tmp_path):
    img, con = _synthetic_figure(tmp_path)
    result = run_pipeline(img, con, str(tmp_path / "out"))
    data = json.loads(result.summary_path.read_text())
    assert data["status"] == "success"
    assert data["n_curves"] == 2
    assert {"line_width_px", "plot_w_px", "plot_h_px"} <= data["metrics"].keys()
    assert len(data["curves"]) == 2
    assert "findings" in data and "n_findings" in data


def test_pipeline_calibrates_physical_values(tmp_path):
    img, con = _synthetic_figure(tmp_path)
    result = run_pipeline(img, con, str(tmp_path / "out"))
    for t in result.tracks:
        caps = [c for c, _ in t.data_raw]
        volts = [v for _, v in t.data_raw]
        assert min(caps) >= -1 and max(caps) <= 321
        assert 1.9 <= min(volts) and max(volts) <= 4.1


def test_pipeline_no_debug_skips_overlay(tmp_path):
    img, con = _synthetic_figure(tmp_path)
    result = run_pipeline(img, con, str(tmp_path / "out"), debug=False)
    assert result.overlay_path is None
