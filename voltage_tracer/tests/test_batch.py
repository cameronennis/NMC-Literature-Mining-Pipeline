import json

import cv2
import numpy as np

from voltage_tracer.batch import run_batch


def _write_figure(path, with_curves=True):
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 20), (360, 260), (0, 0, 0), 2)
    if with_curves:
        xs = np.arange(45, 340)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            cv2.line(img, (x0, int(40 + 0.6 * (x0 - 45))),
                     (x1, int(40 + 0.6 * (x1 - 45))), (220, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _contract_dict():
    return {
        "plot_bbox_px": [40, 20, 360, 260],
        "x_axis": {"scale": "linear", "ticks_px": [40, 360], "ticks_val": [0, 320]},
        "y_axis": {"scale": "linear", "ticks_px": [260, 20], "ticks_val": [2.0, 4.0]},
        "series": [{"id": "red_dis", "color_rgb": [220, 30, 30], "label": "d",
                    "direction": "decreasing"}],
    }


def _setup(tmp_path):
    ind = tmp_path / "in"; cond = tmp_path / "con"; outd = tmp_path / "out"
    ind.mkdir(); cond.mkdir()
    return ind, cond, outd


def test_batch_processes_all_matched_figures(tmp_path):
    ind, cond, outd = _setup(tmp_path)
    for name in ("figA", "figB"):
        _write_figure(ind / f"{name}.png")
        (cond / f"{name}.json").write_text(json.dumps(_contract_dict()))
    summary = run_batch(str(ind), str(cond), str(outd))
    assert summary["n_figures"] == 2
    assert summary["n_success"] == 2 and summary["n_failed"] == 0
    assert (outd / "batch_summary.json").exists()
    assert (outd / "figA__red_dis.csv").exists()


def test_batch_records_missing_contract_without_halting(tmp_path):
    ind, cond, outd = _setup(tmp_path)
    _write_figure(ind / "orphan.png")            # no contract
    _write_figure(ind / "good.png")
    (cond / "good.json").write_text(json.dumps(_contract_dict()))
    summary = run_batch(str(ind), str(cond), str(outd))
    statuses = {r["figure"]: r["status"] for r in summary["results"]}
    assert statuses["orphan"] == "no_contract"
    assert statuses["good"] == "success"
    assert summary["n_no_contract"] == 1 and summary["n_success"] == 1


def test_batch_catches_malformed_contract_and_continues(tmp_path):
    ind, cond, outd = _setup(tmp_path)
    _write_figure(ind / "bad.png")
    (cond / "bad.json").write_text("{ this is not valid json")
    _write_figure(ind / "good.png")
    (cond / "good.json").write_text(json.dumps(_contract_dict()))
    summary = run_batch(str(ind), str(cond), str(outd))
    by = {r["figure"]: r for r in summary["results"]}
    assert by["bad"]["status"] == "failed"
    assert "traceback" in by["bad"] and by["bad"]["error"]
    assert by["good"]["status"] == "success"  # batch survived the bad one


def test_batch_aggregates_findings(tmp_path):
    ind, cond, outd = _setup(tmp_path)
    _write_figure(ind / "f.png")
    (cond / "f.json").write_text(json.dumps(_contract_dict()))
    summary = run_batch(str(ind), str(cond), str(outd))
    assert "total_findings" in summary
    assert summary["total_findings"] == sum(
        r.get("n_findings", 0) for r in summary["results"]
    )


def test_batch_debug_writes_overlays(tmp_path):
    ind, cond, outd = _setup(tmp_path)
    _write_figure(ind / "f.png")
    (cond / "f.json").write_text(json.dumps(_contract_dict()))
    run_batch(str(ind), str(cond), str(outd), debug=True)
    assert (outd / "f__overlay.png").exists()
