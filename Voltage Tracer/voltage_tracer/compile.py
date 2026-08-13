"""Stage: (capacity, voltage) -> (time, areal power). Deterministic, no tokens.

Reads the per-curve trace CSVs written by the pipeline plus the conversion
metadata carried in each figure's contract, and writes one (time, areal_power)
CSV per curve -- raw and smoothed (and resampled) lineage carried forward to
mirror the voltage columns.

The one idea: normalise every curve to a single CANONICAL AREAL basis --
areal capacity (mAh/cm^2) and areal current density j (mA/cm^2) -- then apply
one formula, regardless of how the figure originally expressed things:

    time            t = areal_capacity / j                [h]
    areal power     P = voltage * j                       [mW/cm^2]

(Units check: [mAh/cm^2] / [mA/cm^2] = h; [V] * [mA/cm^2] = mW/cm^2.)

`j` is constant per curve, so time is just a rescaled capacity axis and power
is voltage scaled by `j`. Two bases only -- `specific` (per gram) and `areal`
(per area) -- bridged by the electrode loading (mg/cm^2):

    areal = specific * loading / 1000

Resolving j, per curve:
  * a directly-stated current (`series.current_value` + `current_basis`)   -> bridge to areal;
  * else a C-rate (`series.c_rate`)                                        -> j = c_rate * Q_ref_areal.

Resolving Q_ref_areal (the nominal capacity a C-rate is defined against),
once per panel/contract and reused by every curve:
  * a stated theoretical/nominal capacity (specific or areal)              -> bridge to areal;
  * else FALLBACK: the panel's lowest-C-rate curve with c_rate <= 1, using
    its delivered areal capacity. Tagged low-confidence (the estimate stands
    in for a value the paper never states).

A curve whose j (or whose areal capacity) can't be resolved is SKIPPED with a
recorded reason -- never guessed.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from voltage_tracer.contract import Basis, load_contract

# C-rate at or below which a curve's delivered capacity is a representative
# stand-in for the nominal capacity (a faster curve delivers far below nominal).
FALLBACK_MAX_C_RATE = 1.0


def _to_areal(value: float, basis: Basis, loading_mg_cm2: float | None) -> float | None:
    """Bridge a specific/areal capacity or current to the areal basis.

    Returns None when a specific value can't be bridged for want of a loading.
    """
    if basis is Basis.areal:
        return value
    if loading_mg_cm2 is None:
        return None
    return value * loading_mg_cm2 / 1000.0


@dataclass(frozen=True)
class CurveConversion:
    """Everything needed to convert one curve's (capacity, voltage) trace.

    j_mA_cm2       resolved areal current density (constant along the curve)
    cap_to_areal   multiply an x-axis capacity value by this to get mAh/cm^2
                   (1.0 if the axis is already areal, loading/1000 if specific)
    reference_source  how Q_ref was found: "stated_theoretical",
                   "fallback_slowest_curve", or None (direct current -- no
                   reference needed)
    q_ref_areal_mah_cm2  the reference capacity used (None for direct current)
    low_confidence  True when j leans on the fallback reference
    """

    j_mA_cm2: float
    cap_to_areal: float
    reference_source: str | None
    q_ref_areal_mah_cm2: float | None
    low_confidence: bool


def time_h(areal_capacity_mah_cm2: float, conv: CurveConversion) -> float:
    """t = areal_capacity / j, in hours."""
    return areal_capacity_mah_cm2 / conv.j_mA_cm2


def areal_power_mW_cm2(voltage_v: float, conv: CurveConversion) -> float:
    """P = voltage * areal current density, in mW/cm^2."""
    return voltage_v * conv.j_mA_cm2


def _read_trace(trace_csv: Path) -> list[dict]:
    with trace_csv.open() as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def _delivered_areal_capacity(rows: list[dict], cap_to_areal: float) -> float | None:
    """Max areal capacity reached along a trace (its delivered capacity)."""
    caps = [_num(r, "capacity") for r in rows]
    caps = [c for c in caps if c is not None]
    if not caps:
        return None
    return max(caps) * cap_to_areal


def _effective_loading(series, contract) -> float | None:
    """Per-curve loading if given, else the figure-level loading."""
    return series.loading_mg_cm2 if series.loading_mg_cm2 is not None else contract.loading_mg_cm2


@dataclass(frozen=True)
class PanelReference:
    """The panel's nominal reference capacity for turning C-rates into currents.

    Exactly one of `specific_value` / `areal_value` is set (or neither, when no
    reference is available):
      * specific_value -- a per-gram nominal capacity (mAh/g); each curve bridges
        it to areal with ITS OWN loading, so different-loading curves get
        different areal references (the "various mass loadings" case).
      * areal_value -- a single areal capacity (mAh/cm^2) shared by every curve
        (a stated areal theoretical, or the fallback slowest-curve estimate).
    """

    source: str | None
    specific_value: float | None
    areal_value: float | None
    low_confidence: bool


_NO_REFERENCE = PanelReference(None, None, None, False)


def _resolve_panel_reference(contract, traces_by_id: dict[str, list[dict]]) -> PanelReference:
    """Resolve the panel's reference capacity once, shared by all its curves.

    Stated theoretical wins (kept specific so per-curve loading applies, or areal
    if that's how it's given); otherwise the fallback reads the slowest (<=1C)
    curve's delivered areal capacity as a single shared estimate.
    """
    if contract.theoretical_capacity_value is not None:
        val = contract.theoretical_capacity_value
        if contract.theoretical_capacity_basis is Basis.specific:
            return PanelReference("stated_theoretical", val, None, False)
        return PanelReference("stated_theoretical", None, val, False)

    # Fallback: the lowest-C-rate curve with c_rate <= 1, by delivered capacity.
    best_c = None
    best_areal = None
    for ser in contract.series:
        if ser.c_rate is None or ser.c_rate > FALLBACK_MAX_C_RATE:
            continue
        rows = traces_by_id.get(ser.id)
        if not rows:
            continue
        cap_to_areal = _to_areal(1.0, contract.x_capacity_basis, _effective_loading(ser, contract))
        if cap_to_areal is None:  # specific x with no loading -> can't bridge
            continue
        delivered = _delivered_areal_capacity(rows, cap_to_areal)
        if delivered is None:
            continue
        if best_c is None or ser.c_rate < best_c:
            best_c, best_areal = ser.c_rate, delivered
    if best_areal is not None:
        return PanelReference("fallback_slowest_curve", None, best_areal, True)
    return _NO_REFERENCE


def _curve_conversion(series, contract, ref: PanelReference) -> tuple[CurveConversion | None, str]:
    """Build a CurveConversion for one series, or (None, reason) if unresolvable."""
    loading = _effective_loading(series, contract)
    cap_to_areal = _to_areal(1.0, contract.x_capacity_basis, loading)
    if cap_to_areal is None:
        return None, "x-axis is specific capacity but no loading is available"

    # A directly-stated current takes precedence and needs no reference.
    if series.current_value is not None:
        j = _to_areal(series.current_value, series.current_basis, loading)
        if j is None:
            return None, "specific current stated but no loading to make it areal"
        return CurveConversion(j, cap_to_areal, None, None, False), ""

    if series.c_rate is not None:
        # This curve's areal reference: a stated specific value scales by this
        # curve's own loading; a stated-areal or fallback value is shared as-is.
        if ref.specific_value is not None:
            q_ref = _to_areal(ref.specific_value, Basis.specific, loading)
            if q_ref is None:
                return None, "c_rate with a specific reference but no loading to make it areal"
        elif ref.areal_value is not None:
            q_ref = ref.areal_value
        else:
            return None, "c_rate given but no reference capacity (stated or <=1C fallback)"
        j = series.c_rate * q_ref
        return CurveConversion(j, cap_to_areal, ref.source, q_ref, ref.low_confidence), ""

    return None, "no current: neither current_value nor c_rate"


def _trace_is_bridged(rows: list[dict]) -> bool:
    """Whether a trace CSV contains any occlusion-bridged (reconstructed) point
    -- the per-point `bridged` flag written by the pipeline. Absent column
    (older traces) reads as not bridged."""
    return any(str(r.get("bridged", "")).strip() == "1" for r in rows)


def compile_curve_csv(trace_csv: Path, conv: CurveConversion, out_csv: Path) -> None:
    """Convert one per-curve trace CSV to a (time, areal_power) CSV, keeping
    the raw/smoothed/resampled lineage. Reads the trace's `capacity`,
    `voltage_raw`, `voltage_smooth`, `capacity_uniform`, `voltage_resampled`
    columns and appends two constant per-curve confidence flags:
    `reference_low_confidence` (C-rate reference was the fallback estimate) and
    `bridge_low_confidence` (the trace includes a reconnected occlusion span)."""
    rows = _read_trace(trace_csv)
    ref_flag = 1 if conv.low_confidence else 0
    bridge_flag = 1 if _trace_is_bridged(rows) else 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_h", "power_raw_mW_cm2", "power_smooth_mW_cm2",
            "time_uniform_h", "power_resampled_mW_cm2",
            "reference_low_confidence", "bridge_low_confidence",
        ])
        for r in rows:
            cap, cap_u = _num(r, "capacity"), _num(r, "capacity_uniform")
            vr, vs, vres = _num(r, "voltage_raw"), _num(r, "voltage_smooth"), _num(r, "voltage_resampled")
            t = time_h(cap * conv.cap_to_areal, conv) if cap is not None else ""
            tu = time_h(cap_u * conv.cap_to_areal, conv) if cap_u is not None else ""
            w.writerow([
                t,
                areal_power_mW_cm2(vr, conv) if vr is not None else "",
                areal_power_mW_cm2(vs, conv) if vs is not None else "",
                tu,
                areal_power_mW_cm2(vres, conv) if vres is not None else "",
                ref_flag, bridge_flag,
            ])


def compile_run(traces_dir: str, contracts_dir: str, output_dir: str) -> dict:
    """Compile every per-curve trace CSV in `traces_dir` (named
    `<figstem>__<series_id>.csv`) using its `<figstem>.json` contract. Traces
    are grouped by figure so the panel's reference capacity is resolved once and
    shared by every curve. Writes compiled CSVs to `output_dir` and returns (and
    writes as `compile_manifest.json`) a summary of compiled vs skipped."""
    traces, contracts, out = Path(traces_dir), Path(contracts_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiled: list[dict] = []
    skipped: list[dict] = []

    # Group trace CSVs by figure stem so a panel is compiled as a unit.
    by_fig: dict[str, list[tuple[str, Path]]] = {}
    for csv_path in sorted(traces.glob("*__*.csv")):
        figstem, series_id = csv_path.stem.split("__", 1)
        by_fig.setdefault(figstem, []).append((series_id, csv_path))

    for figstem, curves in by_fig.items():
        con_path = contracts / f"{figstem}.json"
        if not con_path.exists():
            for series_id, csv_path in curves:
                skipped.append({"curve": csv_path.stem, "reason": f"no contract {con_path.name}"})
            continue
        try:
            contract = load_contract(con_path)
        except Exception as exc:  # noqa: BLE001
            for series_id, csv_path in curves:
                skipped.append({"curve": csv_path.stem, "reason": f"bad contract: {exc}"})
            continue

        # Read every curve's trace once; the reference resolution needs them.
        traces_by_id = {sid: _read_trace(p) for sid, p in curves}
        ref = _resolve_panel_reference(contract, traces_by_id)

        for series_id, csv_path in curves:
            stem = csv_path.stem
            ser = next((s for s in contract.series if s.id == series_id), None)
            if ser is None:
                skipped.append({"curve": stem, "reason": f"series {series_id!r} not in contract"})
                continue
            conv, reason = _curve_conversion(ser, contract, ref)
            if conv is None:
                skipped.append({"curve": stem, "reason": reason})
                continue
            compile_curve_csv(csv_path, conv, out / f"{stem}.csv")
            compiled.append({
                "curve": stem,
                "reference_source": conv.reference_source,
                "q_ref_areal_mah_cm2": conv.q_ref_areal_mah_cm2,
                "j_mA_cm2": conv.j_mA_cm2,
                "low_confidence": conv.low_confidence,
                "bridge_low_confidence": _trace_is_bridged(traces_by_id[series_id]),
            })

    manifest = {
        "n_compiled": len(compiled),
        "n_skipped": len(skipped),
        "compiled": compiled,
        "skipped": skipped,
    }
    (out / "compile_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
