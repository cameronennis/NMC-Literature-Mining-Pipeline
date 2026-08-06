"""Pixel -> data-space calibration.

A pure affine (or log-affine) map from image pixels to (capacity, voltage),
fitted once per axis from the contract's tick pairs. Deliberately decoupled
from tracing and smoothing: it changes no topology, only relabels finalized
points, so a calibration error can never alter which curve is which. Both the
raw and the smoothed pixel traces are mapped; the smoothed curve is also
resampled onto a uniform capacity grid for downstream cross-paper averaging.

Tracker points are in CROP-local pixels (the crop starts at the plot box's
top-left), while the contract's ticks are in ORIGINAL image pixels, so the
crop origin is added before applying the transform.
"""
from dataclasses import dataclass

import numpy as np

from voltage_tracer.contract import AxisScale, Contract
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.tracker import Track


@dataclass(frozen=True)
class AxisTransform:
    """Affine map between one axis's pixels and data values. For a log axis
    the fit is linear in log10(value); `to_data`/`to_px` handle the
    transform so callers work in real data units either way."""
    slope: float       # d(fit_value)/d(pixel), fit_value = value or log10(value)
    intercept: float
    scale: AxisScale

    def to_data(self, px: float | np.ndarray):
        v = self.slope * np.asarray(px, dtype=float) + self.intercept
        return np.power(10.0, v) if self.scale is AxisScale.log else v

    def to_px(self, value: float | np.ndarray):
        v = np.asarray(value, dtype=float)
        if self.scale is AxisScale.log:
            v = np.log10(v)
        return (v - self.intercept) / self.slope


def build_affine_transform(
    ticks_px: list[float], ticks_val: list[float], scale: AxisScale = AxisScale.linear
) -> AxisTransform:
    """Least-squares fit of data value vs pixel over ALL tick pairs (more
    robust to an imprecise tick than a 2-point line). Log axes are fitted in
    log10 space and inverted on read. The contract validator has already
    guaranteed >= 2 distinct ticks and positive values on a log axis."""
    px = np.asarray(ticks_px, dtype=float)
    vals = np.asarray(ticks_val, dtype=float)
    if scale is AxisScale.log:
        vals = np.log10(vals)
    slope, intercept = np.polyfit(px, vals, 1)
    return AxisTransform(float(slope), float(intercept), scale)


def build_axes(contract: Contract) -> tuple[AxisTransform, AxisTransform]:
    """Both axis transforms from a contract."""
    return (
        build_affine_transform(
            contract.x_axis.ticks_px, contract.x_axis.ticks_val, contract.x_axis.scale
        ),
        build_affine_transform(
            contract.y_axis.ticks_px, contract.y_axis.ticks_val, contract.y_axis.scale
        ),
    )


def pixelspace_to_dataspace(
    points: list[tuple[float, float]],
    x_transform: AxisTransform,
    y_transform: AxisTransform,
    origin: tuple[int, int] = (0, 0),
) -> list[tuple[float, float]]:
    """Map crop-local pixel points to (capacity, voltage). `origin` is the
    plot box's top-left in original-image pixels, added to each point so the
    contract's original-frame tick calibration applies."""
    ox, oy = origin
    return [
        (float(x_transform.to_data(x + ox)), float(y_transform.to_data(y + oy)))
        for x, y in points
    ]


def resample_uniform(
    data: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Resample a (capacity, voltage) curve onto a uniform capacity grid.

    The grid spans the curve's capacity range with the SAME number of points
    it already has, so the step is derived from the data range and the
    curve's own point density -- no baked grid. Endpoints are preserved.
    Capacity is strictly increasing (columns run left to right), so a plain
    linear interpolation of voltage on the grid is well-defined, including
    down the near-vertical terminal drop (steep but single-valued)."""
    if len(data) < 2:
        return list(data)
    caps = np.array([c for c, _ in data], dtype=float)
    volts = np.array([v for _, v in data], dtype=float)
    order = np.argsort(caps)
    caps, volts = caps[order], volts[order]
    caps, uniq = np.unique(caps, return_index=True)  # strictly increasing for interp
    volts = volts[uniq]
    if caps.size < 2:
        return list(data)
    grid = np.linspace(caps[0], caps[-1], len(data))
    return [(float(c), float(v)) for c, v in zip(grid, np.interp(grid, caps, volts))]


def calibrate_track(
    track: Track,
    x_transform: AxisTransform,
    y_transform: AxisTransform,
    origin: tuple[int, int],
) -> Track:
    """Populate `data_raw`, `data_smooth`, `data_resampled` on a track from
    its pixel traces. Pixel traces are left untouched."""
    track.data_raw = pixelspace_to_dataspace(
        track.points_raw, x_transform, y_transform, origin
    )
    if track.points_smooth is not None:
        track.data_smooth = pixelspace_to_dataspace(
            track.points_smooth, x_transform, y_transform, origin
        )
        track.data_resampled = resample_uniform(track.data_smooth)
    return track


def calibrate_all(
    tracks: list[Track], contract: Contract, metrics: FigureMetrics | None = None
) -> list[Track]:
    """Calibrate every track from the contract's axes and plot-box origin."""
    x_transform, y_transform = build_axes(contract)
    origin = (
        int(round(contract.plot_bbox_px[0])),
        int(round(contract.plot_bbox_px[1])),
    )
    for t in tracks:
        calibrate_track(t, x_transform, y_transform, origin)
    return tracks
