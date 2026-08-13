import csv
import json

import pytest

from voltage_tracer.compile import (
    CurveConversion,
    areal_power_mW_cm2,
    compile_curve_csv,
    compile_run,
    time_h,
)


# --- unit conversions (hand-checked) ---


def test_time_and_power_from_areal():
    # j = 1.4776 mA/cm^2 (the areal current of a 0.2C, 200 mAh/g, 36.94 mg/cm^2
    # curve: 0.2 * 200 * 36.94 / 1000).
    conv = CurveConversion(j_mA_cm2=1.4776, cap_to_areal=36.94 / 1000.0,
                           reference_source="stated_theoretical",
                           q_ref_areal_mah_cm2=200 * 36.94 / 1000.0, low_confidence=False)
    # 100 mAh/g -> areal 100*36.94/1000 = 3.694 mAh/cm^2; t = 3.694 / 1.4776 = 2.5 h
    areal_cap = 100.0 * conv.cap_to_areal
    assert time_h(areal_cap, conv) == pytest.approx(2.5)
    # P = 3.5 V * 1.4776 = 5.1716 mW/cm^2
    assert areal_power_mW_cm2(3.5, conv) == pytest.approx(5.1716)


def test_matches_legacy_specific_c_rate_formula():
    """The canonical areal basis is algebraically identical to the old
    t = capacity/(C*Q1C), P = V*j formula for the specific-x + C-rate case."""
    C, Q1C, load = 0.2, 200.0, 36.94
    j = C * Q1C * load / 1000.0
    conv = CurveConversion(j, load / 1000.0, "stated_theoretical", Q1C * load / 1000.0, False)
    for cap, v in [(100.0, 3.5), (37.0, 3.1), (200.0, 3.7)]:
        assert time_h(cap * conv.cap_to_areal, conv) == pytest.approx(cap / (C * Q1C))
        assert areal_power_mW_cm2(v, conv) == pytest.approx(v * j)


# --- CSV compile ---


def _write_trace(path, cap_max=200.0):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity", "voltage_raw", "voltage_smooth",
                    "capacity_uniform", "voltage_resampled"])
        w.writerow([0.0, 3.0, 3.0, 0.0, 3.0])
        w.writerow([cap_max / 2, 3.5, 3.4, cap_max / 2, 3.45])
        w.writerow([cap_max, 3.7, 3.7, cap_max, 3.7])


def _contract_dict(**over):
    d = {
        "plot_bbox_px": [0, 0, 500, 400],
        "x_axis": {"scale": "linear", "ticks_px": [0, 500], "ticks_val": [0, 200]},
        "y_axis": {"scale": "linear", "ticks_px": [400, 0], "ticks_val": [2.5, 4.5]},
        "series": [{"id": "0.2C_charge", "color_rgb": [0, 0, 0], "label": "chg",
                    "direction": "increasing", "c_rate": 0.2}],
        "loading_mg_cm2": 36.94,
        "theoretical_capacity_value": 200.0,
        "theoretical_capacity_basis": "specific",
    }
    d.update(over)
    return d


def test_compile_curve_csv_values(tmp_path):
    trace = tmp_path / "fig__0.2C_charge.csv"
    _write_trace(trace)
    out = tmp_path / "out.csv"
    conv = CurveConversion(1.4776, 36.94 / 1000.0, "stated_theoretical",
                           200 * 36.94 / 1000.0, False)
    compile_curve_csv(trace, conv, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == [
        "time_h", "power_raw_mW_cm2", "power_smooth_mW_cm2",
        "time_uniform_h", "power_resampled_mW_cm2",
        "reference_low_confidence", "bridge_low_confidence",
    ]
    # row 2: capacity 100 -> t = 2.5 h; power_raw = 3.5 * 1.4776
    assert float(rows[1]["time_h"]) == pytest.approx(2.5)
    assert float(rows[1]["power_raw_mW_cm2"]) == pytest.approx(3.5 * 1.4776)
    assert float(rows[1]["power_smooth_mW_cm2"]) == pytest.approx(3.4 * 1.4776)
    assert rows[1]["reference_low_confidence"] == "0"
    assert rows[1]["bridge_low_confidence"] == "0"  # no bridged column -> 0


def test_bridge_low_confidence_flag_propagates(tmp_path):
    """A trace with any bridged (reconstructed) point flags the whole compiled
    curve, so it can be filtered out in post."""
    trace = tmp_path / "fig__0.2C_charge.csv"
    with trace.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity", "voltage_raw", "voltage_smooth",
                    "capacity_uniform", "voltage_resampled", "bridged"])
        w.writerow([0.0, 3.0, 3.0, 0.0, 3.0, 0])
        w.writerow([100.0, 3.5, 3.4, 100.0, 3.45, 1])  # a bridged point
        w.writerow([200.0, 3.7, 3.7, 200.0, 3.7, 0])
    out = tmp_path / "out.csv"
    conv = CurveConversion(1.4776, 36.94 / 1000.0, "stated_theoretical", 7.388, False)
    compile_curve_csv(trace, conv, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert all(r["bridge_low_confidence"] == "1" for r in rows)  # constant per curve


def _run(tmp_path, contract, traces):
    tdir = tmp_path / "traces"; cdir = tmp_path / "contracts"; out = tmp_path / "compiled"
    tdir.mkdir(); cdir.mkdir()
    for name, cap_max in traces:
        _write_trace(tdir / name, cap_max)
    (cdir / "fig.json").write_text(json.dumps(contract))
    return compile_run(str(tdir), str(cdir), str(out)), out


def test_compile_run_stated_theoretical(tmp_path):
    manifest, out = _run(tmp_path, _contract_dict(), [("fig__0.2C_charge.csv", 200.0)])
    assert manifest["n_compiled"] == 1 and manifest["n_skipped"] == 0
    assert (out / "fig__0.2C_charge.csv").exists()
    entry = manifest["compiled"][0]
    assert entry["reference_source"] == "stated_theoretical"
    assert entry["low_confidence"] is False


def test_compile_run_direct_areal_current(tmp_path):
    """Areal x-axis + a directly-stated areal current: no reference needed."""
    con = _contract_dict(
        x_capacity_basis="areal",
        loading_mg_cm2=None,
        theoretical_capacity_value=None, theoretical_capacity_basis=None,
        series=[{"id": "c1", "color_rgb": [1, 2, 3], "label": "x",
                 "direction": "decreasing", "current_value": 2.0, "current_basis": "areal"}],
    )
    manifest, out = _run(tmp_path, con, [("fig__c1.csv", 5.0)])
    assert manifest["n_compiled"] == 1 and manifest["n_skipped"] == 0
    entry = manifest["compiled"][0]
    assert entry["reference_source"] is None and entry["j_mA_cm2"] == pytest.approx(2.0)
    # areal x: cap 2.5 mAh/cm^2 -> t = 2.5/2.0 = 1.25 h
    with (out / "fig__c1.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[1]["time_h"]) == pytest.approx(1.25)


def test_compile_run_per_curve_loading(tmp_path):
    """'Various mass loadings' shape: each curve carries its own loading."""
    con = _contract_dict(
        loading_mg_cm2=None,  # no figure-level loading
        theoretical_capacity_value=200.0, theoretical_capacity_basis="specific",
        series=[
            {"id": "lowL", "color_rgb": [1, 0, 0], "label": "38", "direction": "decreasing",
             "c_rate": 0.1, "loading_mg_cm2": 38.20},
            {"id": "hiL", "color_rgb": [0, 0, 1], "label": "76", "direction": "decreasing",
             "c_rate": 0.1, "loading_mg_cm2": 76.39},
        ],
    )
    manifest, out = _run(tmp_path, con, [("fig__lowL.csv", 150.0), ("fig__hiL.csv", 150.0)])
    assert manifest["n_compiled"] == 2 and manifest["n_skipped"] == 0
    by = {e["curve"]: e for e in manifest["compiled"]}
    # Stated SPECIFIC theoretical (200 mAh/g) is bridged by EACH curve's own
    # loading, so the areal references (and thus j) differ per curve.
    assert by["fig__lowL"]["reference_source"] == "stated_theoretical"
    assert by["fig__lowL"]["q_ref_areal_mah_cm2"] == pytest.approx(200 * 38.20 / 1000.0)
    assert by["fig__hiL"]["q_ref_areal_mah_cm2"] == pytest.approx(200 * 76.39 / 1000.0)
    assert by["fig__lowL"]["j_mA_cm2"] != by["fig__hiL"]["j_mA_cm2"]
    assert by["fig__lowL"]["low_confidence"] is False


def test_compile_run_fallback_shared_reference(tmp_path):
    """No stated capacity: reference = slowest (<=1C) curve's delivered areal
    capacity, computed once and shared; fallback curves flagged low-confidence."""
    con = _contract_dict(
        theoretical_capacity_value=None, theoretical_capacity_basis=None,
        series=[
            {"id": "slow", "color_rgb": [1, 0, 0], "label": "0.1C", "direction": "decreasing", "c_rate": 0.1},
            {"id": "fast", "color_rgb": [0, 0, 1], "label": "2C", "direction": "decreasing", "c_rate": 2.0},
        ],
    )
    # slowest (0.1C) delivers 200 mAh/g * 36.94/1000 = 7.388 mAh/cm^2 -> Q_ref
    manifest, out = _run(tmp_path, con, [("fig__slow.csv", 200.0), ("fig__fast.csv", 120.0)])
    assert manifest["n_compiled"] == 2
    by = {e["curve"]: e for e in manifest["compiled"]}
    qref = 200.0 * 36.94 / 1000.0
    assert by["fig__slow"]["q_ref_areal_mah_cm2"] == pytest.approx(qref)
    assert by["fig__fast"]["q_ref_areal_mah_cm2"] == pytest.approx(qref)  # shared
    assert by["fig__fast"]["j_mA_cm2"] == pytest.approx(2.0 * qref)
    assert by["fig__slow"]["low_confidence"] is True
    # low-confidence flag stamped in the CSV
    with (out / "fig__fast.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["reference_low_confidence"] == "1"


def test_compile_run_skips_specific_without_loading(tmp_path):
    con = _contract_dict(loading_mg_cm2=None,
                         theoretical_capacity_value=None, theoretical_capacity_basis=None)
    manifest, _ = _run(tmp_path, con, [("fig__0.2C_charge.csv", 200.0)])
    assert manifest["n_compiled"] == 0 and manifest["n_skipped"] == 1
    assert "loading" in manifest["skipped"][0]["reason"]


def test_compile_run_skips_c_rate_without_reference(tmp_path):
    """C-rate curve, no stated capacity, and the only curve is >1C -> no ≤1C
    fallback sibling -> can't resolve, skipped."""
    con = _contract_dict(
        theoretical_capacity_value=None, theoretical_capacity_basis=None,
        series=[{"id": "fast", "color_rgb": [0, 0, 1], "label": "5C",
                 "direction": "decreasing", "c_rate": 5.0}],
    )
    manifest, _ = _run(tmp_path, con, [("fig__fast.csv", 90.0)])
    assert manifest["n_compiled"] == 0 and manifest["n_skipped"] == 1
    assert "reference" in manifest["skipped"][0]["reason"]


def test_compile_run_skips_missing_contract(tmp_path):
    tdir = tmp_path / "traces"; cdir = tmp_path / "contracts"; out = tmp_path / "compiled"
    tdir.mkdir(); cdir.mkdir()
    _write_trace(tdir / "orphan__1st_charge.csv")
    manifest = compile_run(str(tdir), str(cdir), str(out))
    assert manifest["n_skipped"] == 1
    assert "no contract" in manifest["skipped"][0]["reason"]
