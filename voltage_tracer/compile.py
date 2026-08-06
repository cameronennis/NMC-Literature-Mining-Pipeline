"""Stage: (capacity, voltage) -> (time, areal power). Deterministic, no tokens.

Reads the per-curve trace CSVs written by the pipeline plus the conversion
scalars carried in each figure's contract, and writes one (time, areal_power)
CSV per curve -- raw and smoothed (and resampled) lineage carried forward to
mirror the voltage columns.

Per curve, galvanostatic (constant current), with:
  C     = the curve's C-rate            (from series.c_rate)
  Q1C   = 1C / theoretical capacity      (mAh/g, from contract.q_1c_mah_g)
  load  = electrode areal loading        (mg/cm^2, from contract.loading_mg_cm2)

    time            t = capacity / (C * Q1C)            [h]
    areal current   j = C * Q1C * load / 1000           [mA/cm^2]
    areal power     P = voltage * j                     [mW/cm^2]

`j` is constant per curve, so time is just a rescaled capacity axis and power
is voltage scaled by `j`. First-cycle profiles only (same scope rule as the
contract stage). A figure whose contract lacks any needed scalar is SKIPPED
with a recorded reason -- never guessed.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from voltage_tracer.contract import load_contract


@dataclass(frozen=True)
class Scalars:
    c_rate: float
    q_1c_mah_g: float
    loading_mg_cm2: float

    @property
    def areal_current_mA_cm2(self) -> float:
        return self.c_rate * self.q_1c_mah_g * self.loading_mg_cm2 / 1000.0


def capacity_to_time_h(capacity_mah_g: float, s: Scalars) -> float:
    """t = capacity / (C * Q1C), in hours."""
    return capacity_mah_g / (s.c_rate * s.q_1c_mah_g)


def voltage_to_areal_power_mW_cm2(voltage_v: float, s: Scalars) -> float:
    """P = voltage * areal current density, in mW/cm^2."""
    return voltage_v * s.areal_current_mA_cm2


def _missing_scalars(contract, series_id: str) -> list[str]:
    """Names of the scalars needed to compile this curve that are absent."""
    ser = next((x for x in contract.series if x.id == series_id), None)
    missing = []
    if ser is None or ser.c_rate is None:
        missing.append("c_rate")
    if contract.q_1c_mah_g is None:
        missing.append("q_1c_mah_g")
    if contract.loading_mg_cm2 is None:
        missing.append("loading_mg_cm2")
    return missing


def compile_curve_csv(trace_csv: Path, scalars: Scalars, out_csv: Path) -> None:
    """Convert one per-curve trace CSV to a (time, areal_power) CSV, keeping
    the raw/smoothed/resampled lineage. Reads the trace's `capacity`,
    `voltage_raw`, `voltage_smooth`, `capacity_uniform`, `voltage_resampled`
    columns."""
    with trace_csv.open() as f:
        rows = list(csv.DictReader(f))

    def num(row, key):
        v = row.get(key, "")
        return float(v) if v not in ("", None) else None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_h", "power_raw_mW_cm2", "power_smooth_mW_cm2",
            "time_uniform_h", "power_resampled_mW_cm2",
        ])
        for r in rows:
            cap, cap_u = num(r, "capacity"), num(r, "capacity_uniform")
            vr, vs, vres = num(r, "voltage_raw"), num(r, "voltage_smooth"), num(r, "voltage_resampled")
            t = capacity_to_time_h(cap, scalars) if cap is not None else ""
            tu = capacity_to_time_h(cap_u, scalars) if cap_u is not None else ""
            w.writerow([
                t,
                voltage_to_areal_power_mW_cm2(vr, scalars) if vr is not None else "",
                voltage_to_areal_power_mW_cm2(vs, scalars) if vs is not None else "",
                tu,
                voltage_to_areal_power_mW_cm2(vres, scalars) if vres is not None else "",
            ])


def compile_run(traces_dir: str, contracts_dir: str, output_dir: str) -> dict:
    """Compile every per-curve trace CSV in `traces_dir` (named
    `<figstem>__<series_id>.csv`) using scalars from `contracts_dir`'s
    `<figstem>.json`. Writes compiled CSVs to `output_dir` and returns (and
    writes as `compile_manifest.json`) a summary of compiled vs skipped."""
    traces, contracts, out = Path(traces_dir), Path(contracts_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiled: list[str] = []
    skipped: list[dict] = []
    contract_cache: dict[str, object] = {}

    for csv_path in sorted(traces.glob("*__*.csv")):
        stem = csv_path.stem
        figstem, series_id = stem.split("__", 1)
        con_path = contracts / f"{figstem}.json"
        if not con_path.exists():
            skipped.append({"curve": stem, "reason": f"no contract {con_path.name}"})
            continue
        if figstem not in contract_cache:
            try:
                contract_cache[figstem] = load_contract(con_path)
            except Exception as exc:  # noqa: BLE001
                contract_cache[figstem] = exc
        contract = contract_cache[figstem]
        if isinstance(contract, Exception):
            skipped.append({"curve": stem, "reason": f"bad contract: {contract}"})
            continue

        missing = _missing_scalars(contract, series_id)
        if missing:
            skipped.append({"curve": stem, "reason": f"missing scalars: {', '.join(missing)}"})
            continue

        ser = next(x for x in contract.series if x.id == series_id)
        scalars = Scalars(ser.c_rate, contract.q_1c_mah_g, contract.loading_mg_cm2)
        compile_curve_csv(csv_path, scalars, out / f"{stem}.csv")
        compiled.append(stem)

    manifest = {
        "n_compiled": len(compiled),
        "n_skipped": len(skipped),
        "compiled": compiled,
        "skipped": skipped,
    }
    (out / "compile_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
