"""Batch driver: run the pipeline over a directory of figures.

    python -m voltage_tracer.batch --input-dir DIR --contracts-dir DIR \
        --output-dir DIR [--debug]

Pairs each image with its contract by filename stem (``fig.png`` <->
``fig.json``). A failure on one figure (missing/malformed contract, a
tracing error) is caught, recorded with its traceback, and the batch keeps
going -- so one bad paper never halts a 1000-paper run. Writes a single
``batch_summary.json`` listing successes, failures, and every validation
finding for one-pass triage.
"""
import argparse
import json
import traceback
from pathlib import Path

from voltage_tracer.pipeline import run_pipeline

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def run_batch(
    input_dir: str, contracts_dir: str, output_dir: str, debug: bool = False
) -> dict:
    """Trace every figure in `input_dir` against its stem-matched contract in
    `contracts_dir`, writing outputs to `output_dir`. Returns (and writes as
    `batch_summary.json`) the aggregate summary; never raises for a per-figure
    problem."""
    in_dir, con_dir, out_dir = Path(input_dir), Path(contracts_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in in_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    results: list[dict] = []
    for img in images:
        contract = con_dir / f"{img.stem}.json"
        if not contract.exists():
            results.append({"figure": img.stem, "status": "no_contract",
                            "error": f"no contract {contract.name}"})
            continue
        try:
            r = run_pipeline(str(img), str(contract), str(out_dir), debug=debug)
            results.append({
                "figure": r.figure,
                "status": "success",
                "n_curves": len(r.tracks),
                "n_findings": len(r.findings),
                "findings": [
                    {"severity": f.severity, "check": f.check,
                     "curve_id": f.curve_id, "description": f.description,
                     "x_range": list(f.x_range) if f.x_range else None}
                    for f in r.findings
                ],
                "summary": r.summary_path.name,
            })
        except Exception as exc:  # noqa: BLE001 -- deliberately keep the batch alive
            results.append({
                "figure": img.stem,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })

    summary = {
        "n_figures": len(images),
        "n_success": sum(r["status"] == "success" for r in results),
        "n_failed": sum(r["status"] == "failed" for r in results),
        "n_no_contract": sum(r["status"] == "no_contract" for r in results),
        "total_findings": sum(r.get("n_findings", 0) for r in results),
        "results": results,
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, help="directory of figure images")
    ap.add_argument("--contracts-dir", required=True, help="directory of <stem>.json contracts")
    ap.add_argument("--output-dir", required=True, help="where CSVs, summaries, overlays go")
    ap.add_argument("--debug", action="store_true", help="also save a debug overlay per figure")
    args = ap.parse_args()

    summary = run_batch(args.input_dir, args.contracts_dir, args.output_dir, args.debug)
    print(
        f"batch: {summary['n_success']}/{summary['n_figures']} succeeded, "
        f"{summary['n_failed']} failed, {summary['n_no_contract']} without a contract, "
        f"{summary['total_findings']} findings -> {args.output_dir}/batch_summary.json"
    )


if __name__ == "__main__":
    main()
