"""Input JSON contract: axis calibration + series metadata for one figure.

An upstream LLM stage (outside this package) reads a figure and produces
this JSON. This module only defines its schema and validates it -- it does
no image processing itself.
"""
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


class AxisScale(str, Enum):
    linear = "linear"
    log = "log"


class Direction(str, Enum):
    increasing = "increasing"
    decreasing = "decreasing"


class Axis(BaseModel):
    scale: AxisScale
    ticks_px: list[float]
    ticks_val: list[float]

    @model_validator(mode="after")
    def _check_ticks(self) -> "Axis":
        if len(self.ticks_px) != len(self.ticks_val):
            raise ValueError(
                f"ticks_px ({len(self.ticks_px)} entries) and ticks_val "
                f"({len(self.ticks_val)} entries) must be the same length"
            )
        if len(self.ticks_px) < 2:
            raise ValueError(
                f"need at least 2 tick points to calibrate an axis, got "
                f"{len(self.ticks_px)}"
            )
        if len(set(self.ticks_px)) < 2 or len(set(self.ticks_val)) < 2:
            raise ValueError(
                "axis ticks must include at least 2 distinct pixel positions "
                "and 2 distinct values -- identical ticks can't define a "
                "calibration line"
            )
        if self.scale is AxisScale.log and any(v <= 0 for v in self.ticks_val):
            raise ValueError(
                f"log-scale axis tick values must all be positive, got "
                f"{self.ticks_val}"
            )
        return self


class Series(BaseModel):
    id: str
    color_rgb: tuple[int, int, int]
    label: str
    direction: Direction
    c_rate_rank: int | None = None
    # actual C-rate value for this curve (e.g. 0.2 for 0.2C), read from the
    # figure text by the reasoning stage. Needed by compile.py to convert
    # capacity->time and voltage->areal power; a curve missing it is skipped.
    c_rate: float | None = None

    @field_validator("color_rgb")
    @classmethod
    def _check_color_range(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if not all(0 <= c <= 255 for c in v):
            raise ValueError(f"color_rgb values must be in [0, 255], got {v}")
        return v

    @field_validator("c_rate")
    @classmethod
    def _check_c_rate(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"c_rate must be positive, got {v}")
        return v


class Contract(BaseModel):
    plot_bbox_px: tuple[float, float, float, float]
    x_axis: Axis
    y_axis: Axis
    series: list[Series]
    # Optional manual override for the legend region (same [x0, y0, x1, y1]
    # convention as plot_bbox_px, in original-image pixels). When present,
    # preprocess.py masks exactly this region instead of relying on its
    # heuristic legend detection.
    legend_bbox_px: tuple[float, float, float, float] | None = None
    # Optional sub-rectangle of the plot box that curve DATA can occupy,
    # as fractions of the plot box [fx0, fy0, fx1, fy1], measured from the
    # box's top-left in VISUAL orientation (fy0 = top edge, fy1 = bottom
    # edge -- pixel rows, not data values). Anything outside is treated as
    # a no-data region: in-plot titles, panel letters, or a legend that
    # sits over the axes. Curves are known not to enter it, so candidates
    # there are text/artifacts and are dropped, and terminal extensions
    # stop at its edge. Default (whole box) = no restriction. Example: a
    # title across the top and a legend on the right -> [0.0, 0.1, 0.84, 1.0].
    data_region_frac: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    # Figure-level conversion scalars (read from the paper text by the
    # reasoning stage), consumed by compile.py to turn (capacity, voltage)
    # into (time, areal power). q_1c_mah_g = the 1C / theoretical specific
    # capacity; loading_mg_cm2 = the electrode areal loading. Absent => that
    # figure can't be compiled and is skipped (never guessed).
    q_1c_mah_g: float | None = None
    loading_mg_cm2: float | None = None

    @field_validator("q_1c_mah_g", "loading_mg_cm2")
    @classmethod
    def _check_positive_scalar(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"conversion scalar must be positive, got {v}")
        return v

    @field_validator("plot_bbox_px", "legend_bbox_px")
    @classmethod
    def _check_bbox(cls, v):
        if v is None:
            return v
        x0, y0, x1, y1 = v
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"bbox must satisfy x1 > x0 and y1 > y0, got {v}"
            )
        return v

    @field_validator("data_region_frac")
    @classmethod
    def _check_data_region(cls, v):
        fx0, fy0, fx1, fy1 = v
        if not (0.0 <= fx0 < fx1 <= 1.0 and 0.0 <= fy0 < fy1 <= 1.0):
            raise ValueError(
                f"data_region_frac must satisfy 0 <= fx0 < fx1 <= 1 and "
                f"0 <= fy0 < fy1 <= 1, got {v}"
            )
        return v

    @model_validator(mode="after")
    def _check_series(self) -> "Contract":
        if not self.series:
            raise ValueError("contract must declare at least one series")
        ids = [s.id for s in self.series]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate series id(s): {dupes}")
        return self


def validate_contract(data: dict) -> Contract:
    """Validate a raw (already-parsed) dict against the contract schema.

    Raises `pydantic.ValidationError` (which subsumes the `ValueError`s
    raised by the validators above) on malformed input -- this is
    deliberate: bad contracts should fail loudly here, not be guessed at
    downstream.
    """
    return Contract.model_validate(data)


def load_contract(path: str | Path) -> Contract:
    """Read a contract JSON file from disk and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"contract file not found: {path}")
    data = json.loads(path.read_text())
    return validate_contract(data)
