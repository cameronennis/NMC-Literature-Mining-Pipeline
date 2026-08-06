"""Convert pixel-space paths to real data units and to a normalized 0-1
fraction of each axis's configured range."""
import numpy as np


def pixel_to_data(x_px, y_px, bbox, x_min, x_max, y_min, y_max):
    """bbox is (left, right, top, bottom) pixel coords of the plot border,
    as returned by border.detect_plot_border, in the full-image frame.
    x_px/y_px must already be in that same full-image pixel frame."""
    left, right, top, bottom = bbox
    x_data = x_min + (x_px - left) / (right - left) * (x_max - x_min)
    y_data = y_max - (y_px - top) / (bottom - top) * (y_max - y_min)
    return x_data, y_data


def normalize(x_data, y_data, x_min, x_max, y_min, y_max):
    x_norm = (x_data - x_min) / (x_max - x_min)
    y_norm = (y_data - y_min) / (y_max - y_min)
    return np.clip(x_norm, 0, 1), np.clip(y_norm, 0, 1)
