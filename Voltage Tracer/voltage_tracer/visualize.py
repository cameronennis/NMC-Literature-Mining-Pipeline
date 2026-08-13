"""Debug overlays -- the single consolidated visualization surface.

Draws each track's RAW trace as faint points (the measured ground truth) and
its SMOOTHED trace as a bold polyline, plus the auto-detected text/legend
regions as boxes, so any place the smoother departs from the measured points
-- or any region wrongly masked -- is immediately visible. Replaces the
earlier ad-hoc check_*.py scripts.
"""
from pathlib import Path

import cv2
import numpy as np

from voltage_tracer.contract import Contract
from voltage_tracer.metrics import FigureMetrics
from voltage_tracer.tracker import Track

# distinct overlay colors (BGR) cycled across tracks
_TRACK_COLORS = [
    (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0),
    (0, 128, 255), (255, 255, 0), (128, 0, 255), (255, 0, 0),
]


def save_debug_overlay(
    image: np.ndarray,
    contract: Contract,
    metrics: FigureMetrics,
    tracks: list[Track],
    output_path: str | Path,
) -> None:
    """Render raw points + smoothed polylines + detected text regions onto a
    dimmed copy of the original image and write it (as BGR via cv2)."""
    x0 = int(round(contract.plot_bbox_px[0]))
    y0 = int(round(contract.plot_bbox_px[1]))
    overlay = (image * 0.3).astype(np.uint8)

    for i, t in enumerate(tracks):
        color = _TRACK_COLORS[i % len(_TRACK_COLORS)]
        for x, y in t.points_raw:
            cv2.circle(overlay, (int(x) + x0, int(round(y)) + y0), 0, (150, 150, 150), -1)
        smooth = t.points_smooth if t.points_smooth is not None else t.points_raw
        pts = np.array(
            [(int(x) + x0, int(round(y)) + y0) for x, y in smooth], dtype=np.int32
        )
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, color, 1)

    for tx0, ty0, tx1, ty1 in metrics.text_regions:      # label boxes (yellow)
        cv2.rectangle(overlay, (tx0 + x0, ty0 + y0), (tx1 + x0, ty1 + y0),
                      (0, 255, 255), 1)
    for tx0, ty0, tx1, ty1 in metrics.swatch_regions:    # swatches (orange)
        cv2.rectangle(overlay, (tx0 + x0, ty0 + y0), (tx1 + x0, ty1 + y0),
                      (255, 165, 0), 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
