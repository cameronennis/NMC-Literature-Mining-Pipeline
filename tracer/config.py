"""Load per-image (and per-panel, for stacked graphs) axis calibration from
CSV. Each row is one panel: a plain single-graph image has one row; a column
of N stacked graphs has N rows sharing the same filename, one per panel
(numbered top to bottom starting at 1)."""
import pandas as pd

REQUIRED_COLUMNS = ["filename", "x_min", "x_max", "x_unit", "y_min", "y_max", "y_unit"]


def load_config(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Config file {path} is missing columns: {missing}")
    if "panel" not in df.columns:
        df["panel"] = 1
    df["panel"] = df["panel"].fillna(1).astype(int)

    rows_by_file = {}
    for filename, group in df.groupby("filename"):
        rows_by_file[filename] = {
            int(row["panel"]): row.to_dict() for _, row in group.iterrows()
        }
    return rows_by_file
