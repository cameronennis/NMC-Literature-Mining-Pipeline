import csv
import json

import pytest

from voltage_tracer.compile import (
    Scalars,
    capacity_to_time_h,
    compile_curve_csv,
    compile_run,
    voltage_to_areal_power_mW_cm2,
)


# --- unit conversions (hand-checked) ---


def test_time_conversion():
    s = Scalars(c_rate=0.2, q_1c_mah_g=200.0, loading_mg_cm2=36.94)
    # at 0.2C, full 200 mAh/g takes 5 h; 100 mAh/g -> 2.5 h
    assert capacity_to_time_h(100.0, s) == pytest.approx(2.5)
    assert capacity_to_time_h(0.0, s) == pytest.approx(0.0)


def test_areal_current_and_power():
    s = Scalars(c_rate=0.2, q_1c_mah_g=200.0, loading_mg_cm2=36.94)
    # j = 0.2 * 200 * 36.94 / 1000 = 1.4776 mA/cm^2
    assert s.areal_current_mA_cm2 == pytest.approx(1.4776)
    # P = 3.5 V * 1.4776 = 5.1716 mW/cm^2
    assert voltage_to_areal_power_mW_cm2(3.5, s) == pytest.approx(5.1716)


# --- CSV compile ---


def _write_trace(path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity", "voltage_raw", "voltage_smooth",
                    "capacity_uniform", "voltage_resampled"])
        w.writerow([0.0, 3.0, 3.0, 0.0, 3.0])
        w.writerow([100.0, 3.5, 3.4, 100.0, 3.45])
        w.writerow([200.0, 3.7, 3.7, 200.0, 3.7])


def _contract_dict(with_scalars=True):
    d = {
        "plot_bbox_px": [0, 0, 500, 400],
        "x_axis": {"scale": "linear", "ticks_px": [0, 500], "ticks_val": [0, 200]},
        "y_axis": {"scale": "linear", "ticks_px": [400, 0], "ticks_val": [2.5, 4.5]},
        "series": [{"id": "1st_charge", "color_rgb": [0, 0, 0], "label": "chg",
                    "direction": "increasing"}],
    }
    if with_scalars:
        d["series"][0]["c_rate"] = 0.2
        d["q_1c_mah_g"] = 200.0
        d["loading_mg_cm2"] = 36.94
    return d


def test_compile_curve_csv_values(tmp_path):
    trace = tmp_path / "fig__1st_charge.csv"
    _write_trace(trace)
    out = tmp_path / "out.csv"
    s = Scalars(0.2, 200.0, 36.94)
    compile_curve_csv(trace, s, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == [
        "time_h", "power_raw_mW_cm2", "power_smooth_mW_cm2",
        "time_uniform_h", "power_resampled_mW_cm2",
    ]
    # row 2: capacity 100 -> t = 2.5 h; power_raw = 3.5 * 1.4776
    assert float(rows[1]["time_h"]) == pytest.approx(2.5)
    assert float(rows[1]["power_raw_mW_cm2"]) == pytest.approx(3.5 * 1.4776)
    assert float(rows[1]["power_smooth_mW_cm2"]) == pytest.approx(3.4 * 1.4776)


def test_compile_run_compiles_with_scalars(tmp_path):
    traces = tmp_path / "traces"; contracts = tmp_path / "contracts"; out = tmp_path / "compiled"
    traces.mkdir(); contracts.mkdir()
    _write_trace(traces / "fig__1st_charge.csv")
    (contracts / "fig.json").write_text(json.dumps(_contract_dict(with_scalars=True)))
    manifest = compile_run(str(traces), str(contracts), str(out))
    assert manifest["n_compiled"] == 1 and manifest["n_skipped"] == 0
    assert (out / "fig__1st_charge.csv").exists()
    assert (out / "compile_manifest.json").exists()


def test_compile_run_skips_missing_scalars_with_reason(tmp_path):
    traces = tmp_path / "traces"; contracts = tmp_path / "contracts"; out = tmp_path / "compiled"
    traces.mkdir(); contracts.mkdir()
    _write_trace(traces / "fig__1st_charge.csv")
    (contracts / "fig.json").write_text(json.dumps(_contract_dict(with_scalars=False)))
    manifest = compile_run(str(traces), str(contracts), str(out))
    assert manifest["n_compiled"] == 0 and manifest["n_skipped"] == 1
    reason = manifest["skipped"][0]["reason"]
    assert "missing scalars" in reason
    assert "c_rate" in reason and "q_1c_mah_g" in reason and "loading_mg_cm2" in reason
    assert not (out / "fig__1st_charge.csv").exists()  # never writes a guessed file


def test_compile_run_skips_missing_contract(tmp_path):
    traces = tmp_path / "traces"; contracts = tmp_path / "contracts"; out = tmp_path / "compiled"
    traces.mkdir(); contracts.mkdir()
    _write_trace(traces / "orphan__1st_charge.csv")
    manifest = compile_run(str(traces), str(contracts), str(out))
    assert manifest["n_skipped"] == 1
    assert "no contract" in manifest["skipped"][0]["reason"]
