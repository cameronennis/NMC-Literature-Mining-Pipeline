#!/usr/bin/env python3
"""Stage 2: turn Stage 1's traced voltage-vs-capacity data into areal power
vs. time, and group it by shared parameters for cross-paper comparison.

Deliberately has NO PDF/CV/OCR dependencies (no fitz, cv2, pytesseract) --
just pandas/numpy -- so it's cheap to re-run as often as you like while
iterating on grouping choices or fixing a conversion bug, without ever
touching the source PDFs again.

Usage:
    python compile.py raw/ --outdir final/
    python compile.py raw/ --outdir final/ --group-by cathode_composition,temperature_C
"""
import argparse
import os

import numpy as np
import pandas as pd

# "specific" capacity (per mass) needs the electrode's areal loading to
# become "areal" capacity (per area); this is the same formula used during
# current-density resolution in metadata_extract.py.
_SPECIFIC_UNITS = {"mah/g", "mahg-1", "mah g-1"}
_AREAL_UNITS = {"mahcm-2", "mahcm2", "mah/cm2", "mah cm-2"}


def _normalize_unit(u):
    if not isinstance(u, str):
        return ""
    return u.replace(" ", "").replace("−", "-").lower()


def load_stage1(raw_dir):
    metadata = pd.read_csv(os.path.join(raw_dir, "metadata.csv"))
    traces = pd.read_csv(os.path.join(raw_dir, "traces.csv"))
    return metadata, traces


def _areal_capacity(x_data, x_unit, areal_loading_mg_cm2):
    unit = _normalize_unit(x_unit)
    if unit in _AREAL_UNITS:
        return x_data, True
    if unit in _SPECIFIC_UNITS and pd.notna(areal_loading_mg_cm2):
        return x_data * float(areal_loading_mg_cm2) / 1000.0, True
    return x_data, False


def compute_power_time(metadata, traces):
    """Returns a long/tidy DataFrame: curve_id, time_h, power_mW_cm2 --
    only for curves with a resolved current density. Charge and discharge
    are two halves of one continuous galvanostatic experiment: discharge's
    time axis picks up where charge's leaves off, rather than each
    restarting at t=0."""
    meta = metadata.set_index("curve_id")
    rows = []

    # Group by everything except direction, so a charge/discharge pair can
    # be concatenated into one continuous time series.
    group_cols = ["pdf", "page", "figure_index", "panel", "color_name"]
    for col in group_cols:
        if col not in meta.columns:
            meta[col] = ""

    for _, group in meta.reset_index().groupby(group_cols, dropna=False):
        group = group.sort_values(
            "direction", key=lambda s: s.map({"charge": 0, "discharge": 1}).fillna(2))
        t_offset = 0.0
        for _, row in group.iterrows():
            curve_id = row["curve_id"]
            current_density = row.get("current_density_mA_cm2")
            if pd.isna(current_density) or current_density in ("", None):
                continue
            current_density = float(current_density)
            if current_density == 0:
                continue

            pts = traces[traces["curve_id"] == curve_id].sort_values("x_px")
            if pts.empty:
                continue

            areal_capacity, ok = _areal_capacity(
                pts["x_data"].to_numpy(), row.get("x_unit"), row.get("areal_loading_mg_cm2"))
            if not ok:
                # capacity axis unit isn't areal and can't be converted --
                # skip rather than silently mis-scale.
                continue

            t_h = areal_capacity / abs(current_density) + t_offset
            power = pts["y_data"].to_numpy() * abs(current_density)  # V x mA/cm2 = mW/cm2

            rows.append(pd.DataFrame({
                "curve_id": curve_id, "time_h": t_h, "power_mW_cm2": power,
            }))
            t_offset = float(np.nanmax(t_h))

    if not rows:
        return pd.DataFrame(columns=["curve_id", "time_h", "power_mW_cm2"])
    return pd.concat(rows, ignore_index=True)


def group_and_save(power_time, metadata, group_by, outdir):
    os.makedirs(outdir, exist_ok=True)
    power_time.to_csv(os.path.join(outdir, "power_time.csv"), index=False)

    if not group_by:
        return
    merged = power_time.merge(metadata[["curve_id"] + group_by], on="curve_id", how="left")
    before = merged["curve_id"].nunique()
    merged = merged.dropna(subset=group_by)
    after = merged["curve_id"].nunique()
    if after < before:
        print(f"Note: {before - after} curve(s) skipped from grouping "
              f"(missing one or more of {group_by}); still present in power_time.csv.")

    group_summary = os.path.join(outdir, "groups.csv")
    groups = merged.groupby(group_by)["curve_id"].nunique().reset_index(name="n_curves")
    groups.to_csv(group_summary, index=False)
    print(f"Grouped by {group_by}: {len(groups)} distinct group(s) -> {group_summary}")

    grouped_dir = os.path.join(outdir, "grouped")
    os.makedirs(grouped_dir, exist_ok=True)
    for key, sub in merged.groupby(group_by):
        key_str = "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key_str)
        sub.drop(columns=group_by).to_csv(os.path.join(grouped_dir, f"{safe}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", help="Stage 1 output directory (contains metadata.csv, traces.csv)")
    parser.add_argument("--outdir", default="final")
    parser.add_argument("--group-by", default=None,
                        help="Comma-separated metadata.csv column names to group by")
    args = parser.parse_args()

    metadata, traces = load_stage1(args.raw_dir)
    power_time = compute_power_time(metadata, traces)
    print(f"Resolved power/time for {power_time['curve_id'].nunique()} of "
          f"{metadata['curve_id'].nunique()} curves.")

    group_by = [c.strip() for c in args.group_by.split(",")] if args.group_by else None
    group_and_save(power_time, metadata, group_by, args.outdir)
    print(f"Done. Output in {args.outdir}/")


if __name__ == "__main__":
    main()
