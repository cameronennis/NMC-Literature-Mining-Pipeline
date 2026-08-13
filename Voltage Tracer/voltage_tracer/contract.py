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


class Basis(str, Enum):
    """Whether a capacity/current quantity is per-gram or per-area.

    The compile stage normalises everything to an AREAL basis (mAh/cm^2,
    mA/cm^2). `specific` values are bridged to areal by the electrode mass
    loading (mg/cm^2): areal = specific * loading / 1000. `absolute` (mAh, mA)
    is deliberately NOT supported -- it isn't expected in this corpus and would
    need an electrode-area bridge for no payoff.
    """

    specific = "specific"  # per gram: mAh/g, mA/g
    areal = "areal"        # per area: mAh/cm^2, mA/cm^2


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
    # figure text by the reasoning stage. compile.py turns it into an areal
    # current via the panel's reference capacity: j = c_rate * Q_ref_areal.
    c_rate: float | None = None
    # A directly-stated current for this curve, an ALTERNATIVE to c_rate (e.g.
    # a figure labelling curves "0.5 mA/cm2", "1 mA/cm2"). `current_basis` says
    # whether current_value is per-gram (mA/g) or per-area (mA/cm2). When given,
    # compile.py uses it in preference to c_rate and needs no reference capacity.
    current_value: float | None = None
    current_basis: Basis | None = None
    # Per-curve electrode loading (mg/cm^2), overriding the figure-level
    # `Contract.loading_mg_cm2`. For "various mass loadings" figures where each
    # curve is a different-loading cell; leave null to inherit the figure value.
    loading_mg_cm2: float | None = None

    @field_validator("color_rgb")
    @classmethod
    def _check_color_range(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if not all(0 <= c <= 255 for c in v):
            raise ValueError(f"color_rgb values must be in [0, 255], got {v}")
        return v

    @field_validator("c_rate", "current_value", "loading_mg_cm2")
    @classmethod
    def _check_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"value must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _check_current_pairing(self) -> "Series":
        if self.current_value is not None and self.current_basis is None:
            raise ValueError(
                f"series {self.id!r} gives current_value={self.current_value} "
                "but no current_basis (specific/areal)"
            )
        return self


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
    # --- Conversion metadata (canonical AREAL basis; see compile.py) ---------
    # What the x-axis capacity is measured in. `specific` (mAh/g) is bridged to
    # areal via `loading_mg_cm2`; `areal` (mAh/cm^2) is already canonical.
    x_capacity_basis: Basis = Basis.specific
    # Figure-level electrode areal loading (mg/cm^2), the bridge specific<->areal.
    # A per-series `loading_mg_cm2` overrides this for that curve. Read from the
    # paper text by the reasoning stage; may be filled from the per-paper pass.
    loading_mg_cm2: float | None = None
    # Reference / nominal capacity that a C-rate is defined against, used to turn
    # a C-rate into an areal current (j = c_rate * Q_ref_areal). `basis` says
    # whether the value is specific (mAh/g, bridged by loading) or areal
    # (mAh/cm^2). When absent, compile.py falls back to the panel's slowest
    # (<=1C) curve's delivered capacity. NOT named for "1C": it's the nominal
    # capacity, however the paper phrases it.
    theoretical_capacity_value: float | None = None
    theoretical_capacity_basis: Basis | None = None

    @field_validator("loading_mg_cm2", "theoretical_capacity_value")
    @classmethod
    def _check_positive_scalar(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"conversion scalar must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _check_theoretical_pairing(self) -> "Contract":
        if (
            self.theoretical_capacity_value is not None
            and self.theoretical_capacity_basis is None
        ):
            raise ValueError(
                "theoretical_capacity_value given without "
                "theoretical_capacity_basis (specific/areal)"
            )
        return self

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
