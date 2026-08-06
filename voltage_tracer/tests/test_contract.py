import json

import pytest
from pydantic import ValidationError

from voltage_tracer.contract import Contract, load_contract, validate_contract


def _valid_data() -> dict:
    return {
        "plot_bbox_px": [10, 20, 500, 400],
        "x_axis": {
            "scale": "linear",
            "ticks_px": [10, 250, 500],
            "ticks_val": [0, 100, 200],
        },
        "y_axis": {
            "scale": "linear",
            "ticks_px": [400, 20],
            "ticks_val": [2.5, 4.5],
        },
        "series": [
            {
                "id": "s1",
                "color_rgb": [255, 0, 0],
                "label": "0.5C charge",
                "direction": "increasing",
                "c_rate_rank": 1,
            },
            {
                "id": "s2",
                "color_rgb": [255, 0, 0],
                "label": "0.5C discharge",
                "direction": "decreasing",
            },
        ],
    }


def test_validate_contract_accepts_valid_data():
    contract = validate_contract(_valid_data())
    assert isinstance(contract, Contract)
    assert len(contract.series) == 2
    assert contract.series[0].c_rate_rank == 1
    assert contract.series[1].c_rate_rank is None


def test_load_contract_reads_and_validates_file(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_valid_data()))
    contract = load_contract(path)
    assert isinstance(contract, Contract)
    assert contract.plot_bbox_px == (10, 20, 500, 400)


def test_load_contract_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_contract(tmp_path / "does_not_exist.json")


def test_load_contract_malformed_json_raises(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_contract(path)


def test_bbox_x1_must_exceed_x0():
    data = _valid_data()
    data["plot_bbox_px"] = [500, 20, 10, 400]
    with pytest.raises(ValidationError, match="x1 > x0"):
        validate_contract(data)


def test_bbox_y1_must_exceed_y0():
    data = _valid_data()
    data["plot_bbox_px"] = [10, 400, 500, 20]
    with pytest.raises(ValidationError, match="y1 > y0"):
        validate_contract(data)


def test_axis_requires_at_least_two_ticks():
    data = _valid_data()
    data["x_axis"]["ticks_px"] = [10]
    data["x_axis"]["ticks_val"] = [0]
    with pytest.raises(ValidationError, match="at least 2 tick points"):
        validate_contract(data)


def test_axis_tick_arrays_must_match_length():
    data = _valid_data()
    data["x_axis"]["ticks_px"] = [10, 250, 500]
    data["x_axis"]["ticks_val"] = [0, 200]
    with pytest.raises(ValidationError, match="must be the same length"):
        validate_contract(data)


def test_series_direction_must_be_valid_enum_value():
    data = _valid_data()
    data["series"][0]["direction"] = "sideways"
    with pytest.raises(ValidationError):
        validate_contract(data)


def test_series_ids_must_be_unique():
    data = _valid_data()
    data["series"][1]["id"] = "s1"
    with pytest.raises(ValidationError, match="duplicate series id"):
        validate_contract(data)


def test_series_must_be_non_empty():
    data = _valid_data()
    data["series"] = []
    with pytest.raises(ValidationError, match="at least one series"):
        validate_contract(data)


def test_color_rgb_must_be_in_byte_range():
    data = _valid_data()
    data["series"][0]["color_rgb"] = [300, 0, 0]
    with pytest.raises(ValidationError, match=r"\[0, 255\]"):
        validate_contract(data)


def test_conversion_scalars_optional_and_validated():
    data = _valid_data()
    c = validate_contract(data)
    assert c.q_1c_mah_g is None and c.loading_mg_cm2 is None
    assert c.series[0].c_rate is None
    data["q_1c_mah_g"] = 200.0
    data["loading_mg_cm2"] = 36.94
    data["series"][0]["c_rate"] = 0.2
    c = validate_contract(data)
    assert c.q_1c_mah_g == 200.0 and c.loading_mg_cm2 == 36.94
    assert c.series[0].c_rate == 0.2


def test_conversion_scalars_reject_nonpositive():
    for field, val in [("q_1c_mah_g", 0), ("loading_mg_cm2", -1)]:
        data = _valid_data()
        data[field] = val
        with pytest.raises(ValidationError, match="must be positive"):
            validate_contract(data)
    data = _valid_data()
    data["series"][0]["c_rate"] = -0.2
    with pytest.raises(ValidationError, match="c_rate must be positive"):
        validate_contract(data)


def test_log_scale_axis_is_accepted():
    data = _valid_data()
    data["x_axis"]["scale"] = "log"
    data["x_axis"]["ticks_val"] = [0.1, 1, 10]
    contract = validate_contract(data)
    assert contract.x_axis.scale.value == "log"


def test_log_scale_axis_rejects_nonpositive_tick_values():
    data = _valid_data()
    data["x_axis"]["scale"] = "log"  # ticks_val includes 0
    with pytest.raises(ValidationError, match="must all be positive"):
        validate_contract(data)


def test_axis_rejects_all_identical_tick_pixels():
    data = _valid_data()
    data["x_axis"]["ticks_px"] = [250, 250, 250]
    with pytest.raises(ValidationError, match="distinct"):
        validate_contract(data)


def test_axis_rejects_all_identical_tick_values():
    data = _valid_data()
    data["x_axis"]["ticks_val"] = [100, 100, 100]
    with pytest.raises(ValidationError, match="distinct"):
        validate_contract(data)


def test_data_region_defaults_to_full_box():
    assert validate_contract(_valid_data()).data_region_frac == (0.0, 0.0, 1.0, 1.0)


def test_data_region_accepts_valid_subrectangle():
    data = _valid_data()
    data["data_region_frac"] = [0.0, 0.1, 0.84, 1.0]
    assert validate_contract(data).data_region_frac == (0.0, 0.1, 0.84, 1.0)


def test_data_region_rejects_inverted_or_out_of_range():
    for bad in ([0.5, 0.1, 0.4, 1.0], [0.0, 0.0, 1.0, 1.2], [-0.1, 0.0, 1.0, 1.0]):
        data = _valid_data()
        data["data_region_frac"] = bad
        with pytest.raises(ValidationError, match="data_region_frac"):
            validate_contract(data)


def test_legend_bbox_is_optional_and_validated():
    data = _valid_data()
    assert validate_contract(data).legend_bbox_px is None
    data["legend_bbox_px"] = [400, 30, 490, 90]
    assert validate_contract(data).legend_bbox_px == (400, 30, 490, 90)
    data["legend_bbox_px"] = [490, 30, 400, 90]
    with pytest.raises(ValidationError, match="x1 > x0"):
        validate_contract(data)
