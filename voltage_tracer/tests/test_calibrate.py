import numpy as np
import pytest

from voltage_tracer.calibrate import (
    build_affine_transform,
    build_axes,
    calibrate_all,
    calibrate_track,
    pixelspace_to_dataspace,
    resample_uniform,
)
from voltage_tracer.contract import AxisScale, validate_contract
from voltage_tracer.tracker import Track


def _contract():
    return validate_contract({
        "plot_bbox_px": [100, 50, 600, 450],
        # x: pixel 100->0 mAh, 600->300 mAh  (0.6 px per mAh)
        "x_axis": {"scale": "linear", "ticks_px": [100, 350, 600],
                   "ticks_val": [0, 150, 300]},
        # y: pixel 450->2.5 V (bottom), 50->4.5 V (top)
        "y_axis": {"scale": "linear", "ticks_px": [450, 250, 50],
                   "ticks_val": [2.5, 3.5, 4.5]},
        "series": [
            {"id": "s", "color_rgb": [255, 0, 0], "label": "d", "direction": "decreasing"},
        ],
    })


# --- affine fit ---


def test_linear_transform_maps_ticks_exactly():
    t = build_affine_transform([100, 350, 600], [0, 150, 300])
    assert t.to_data(100) == pytest.approx(0.0, abs=1e-9)
    assert t.to_data(600) == pytest.approx(300.0)
    assert t.to_data(350) == pytest.approx(150.0)
    # inverse
    assert t.to_px(150.0) == pytest.approx(350.0)


def test_transform_least_squares_robust_to_one_noisy_tick():
    # 4 clean ticks + 1 slightly off; fit should stay close to the clean line
    t = build_affine_transform([0, 100, 200, 300, 400],
                               [0, 10, 20, 30.7, 40])  # 4th tick off by 0.7
    assert t.to_data(0) == pytest.approx(0.0, abs=0.3)
    assert t.to_data(400) == pytest.approx(40.0, abs=0.3)


def test_log_axis_fits_and_inverts():
    # pixels 0,100,200 -> values 0.1, 1, 10 (decades evenly spaced)
    t = build_affine_transform([0, 100, 200], [0.1, 1, 10], scale=AxisScale.log)
    assert t.to_data(0) == pytest.approx(0.1)
    assert t.to_data(100) == pytest.approx(1.0)
    assert t.to_data(200) == pytest.approx(10.0)
    assert t.to_px(1.0) == pytest.approx(100.0)


# --- pixel -> data with crop origin ---


def test_pixelspace_applies_crop_origin():
    xt, yt = build_axes(_contract())
    # crop-local (0, 400) with origin (100, 50) -> original (100, 450) -> (0 mAh, 2.5 V)
    out = pixelspace_to_dataspace([(0, 400)], xt, yt, origin=(100, 50))
    assert out[0][0] == pytest.approx(0.0, abs=1e-9)
    assert out[0][1] == pytest.approx(2.5)


def test_calibrate_track_maps_raw_and_smooth_and_resamples():
    contract = _contract()
    # a discharge in crop-local px: capacity 0->300 as x 0->500, voltage falls
    xs = np.arange(0, 501, 10)
    pts = [(int(x), float(50 + 0.6 * x)) for x in xs]  # y grows downward (voltage falls)
    t = Track("t", "s", "decreasing", points_raw=pts, pixel_sign=+1)
    t.points_smooth = list(pts)
    calibrate_all([t], contract)
    assert t.data_raw is not None and t.data_smooth is not None
    assert len(t.data_raw) == len(pts)
    # first point: crop (0,50)+origin(100,50)=(100,100)-> cap 0, voltage=4.5-(100-50)/200*... check bounds
    cap0, v0 = t.data_raw[0]
    assert cap0 == pytest.approx(0.0, abs=1e-6)
    # capacities strictly increasing
    caps = [c for c, _ in t.data_raw]
    assert all(b > a for a, b in zip(caps, caps[1:]))
    # resampled onto uniform grid, same count, endpoints preserved
    assert t.data_resampled is not None
    rc = [c for c, _ in t.data_resampled]
    steps = np.diff(rc)
    assert np.allclose(steps, steps[0])
    assert rc[0] == pytest.approx(caps[0]) and rc[-1] == pytest.approx(caps[-1])


# --- resampling ---


def test_resample_uniform_grid_and_interpolation():
    # a straight line voltage = 4 - 0.005*cap, non-uniform capacity spacing
    data = [(0.0, 4.0), (10.0, 3.95), (11.0, 3.945), (50.0, 3.75), (100.0, 3.5)]
    out = resample_uniform(data)
    caps = np.array([c for c, _ in out])
    volts = np.array([v for _, v in out])
    assert np.allclose(np.diff(caps), np.diff(caps)[0])  # uniform
    assert caps[0] == pytest.approx(0.0) and caps[-1] == pytest.approx(100.0)
    # interpolated voltages lie on the known line
    assert np.allclose(volts, 4.0 - 0.005 * caps, atol=1e-6)


def test_resample_handles_duplicate_capacities():
    data = [(0.0, 4.0), (5.0, 3.9), (5.0, 3.8), (10.0, 3.5)]  # a duplicate cap
    out = resample_uniform(data)
    assert len(out) == len(data)
    caps = [c for c, _ in out]
    assert all(b >= a for a, b in zip(caps, caps[1:]))


# --- decoupling / dual resolution ---


def test_calibration_changes_no_topology():
    contract = _contract()
    pts = [(int(x), float(50 + 0.5 * x)) for x in range(0, 500, 5)]
    t = Track("t", "s", "decreasing", points_raw=pts, pixel_sign=+1)
    t.points_smooth = list(pts)
    calibrate_all([t], contract)
    assert t.points_raw == pts  # untouched
    assert len(t.data_raw) == len(t.points_raw)


def test_dual_resolution_calibration_matches_in_data_space():
    """Same physical figure at 1x and 2x pixel scale must calibrate to the
    same (capacity, voltage) -- the affine + data-derived resample carry no
    pixel literal."""
    results = {}
    for scale in (1, 2):
        contract = validate_contract({
            "plot_bbox_px": [0, 0, 500 * scale, 400 * scale],
            "x_axis": {"scale": "linear",
                       "ticks_px": [0, 500 * scale], "ticks_val": [0, 250]},
            "y_axis": {"scale": "linear",
                       "ticks_px": [400 * scale, 0], "ticks_val": [2.5, 4.5]},
            "series": [{"id": "s", "color_rgb": [255, 0, 0], "label": "d",
                        "direction": "decreasing"}],
        })
        xs = np.arange(0, 500 * scale, 5 * scale)
        pts = [(int(x), float(0.5 * x)) for x in xs]
        t = Track("t", "s", "decreasing", points_raw=pts, pixel_sign=+1)
        t.points_smooth = list(pts)
        calibrate_all([t], contract)
        # sample voltage at 10 shared capacity fractions
        caps = np.array([c for c, _ in t.data_resampled])
        volts = np.array([v for _, v in t.data_resampled])
        results[scale] = np.interp(np.linspace(caps[0], caps[-1], 10), caps, volts)
    assert np.allclose(results[1], results[2], atol=1e-6)
