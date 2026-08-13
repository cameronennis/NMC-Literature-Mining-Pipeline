"""Orchestrator: PDFs -> figures -> contracts -> traces -> compiled curves.

Wires the four stages into one corpus run and one CLI:

    python -m voltage_tracer.run --pdf-dir PAPERS --out-dir RUN [--model M] \
        [--fallback-model M|none] [--limit N] [--no-reason] [--debug]

Stage map (each writes its own manifest so you can triage without re-running):

    intake   PDFs           -> RUN/figures/   (<stem>.png + <stem>.txt)   [free]
    reason   figures        -> RUN/contracts/ (<stem>.json)               [TOKENS]
    batch    figures+contracts -> RUN/traces/ (<stem>__<series>.csv)      [free]
    compile  traces+contracts  -> RUN/compiled/(<stem>__<series>.csv)     [free]

`reason` is the ONLY stage that costs tokens -- it is the injected-client
Anthropic call. Everything else is deterministic. `--no-reason` reuses
contracts already in RUN/contracts (re-run trace/compile without re-paying),
and `--limit N` caps how many figures reason processes for a cheap trial.

The Anthropic client is injected into `run_reason`/`run_corpus`, so the whole
orchestrator is testable with a mock and the test suite never spends a token
or needs a key. `build_client()` / `main()` construct the real SDK client, and
only when the reason stage will actually run.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from pathlib import Path

from voltage_tracer.batch import run_batch
from voltage_tracer.compile import compile_run
from voltage_tracer.contract import validate_contract
from voltage_tracer.intake import intake_run
from voltage_tracer.reason import (
    DEFAULT_CERTAINTY_THRESHOLD,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    PaperScalars,
    reason_figure,
    reason_paper_scalars,
)

_MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _prune_to(directory: Path, keep_figstems: set[str]) -> list[str]:
    """Delete files in `directory` that don't belong to any kept figure.

    A file's figure is its name up to the first '__' (so `P9_fig6.png`,
    `P9_fig6.json`, and `P9_fig6__1st_charge.csv` all map to `P9_fig6`).
    Manifests/summaries and subdirectories are left alone. Returns the names
    removed. Used by the `--fresh` reconcile path to clear stale artifacts
    from a prior run without touching files this run will (re)produce.
    """
    if not directory.is_dir():
        return []
    removed = []
    for f in directory.iterdir():
        if f.is_dir() or f.name.endswith("manifest.json") or f.name == "batch_summary.json":
            continue
        figstem = f.stem.split("__", 1)[0]
        if figstem not in keep_figstems:
            f.unlink()
            removed.append(f.name)
    return removed


def _link_or_copy(src: Path, dst: Path) -> None:
    """Give a panel its own image file cheaply -- a hardlink to the figure's
    bytes (no extra disk), falling back to a copy across filesystems."""
    if dst == src:
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _paper_of(figstem: str) -> str:
    """Paper id for a figure stem: 'P11_fig3' -> 'P11', 'P10_SI_figS3' ->
    'P10_SI'. Falls back to the whole stem if there's no '_fig' marker."""
    return figstem.rsplit("_fig", 1)[0] if "_fig" in figstem else figstem


def _paper_text(images: list[Path]) -> str:
    """Assemble one paper's text for scalar extraction from its figures'
    `.txt` sidecars (each = its caption + the identical doc-wide scalar block).
    Lines are de-duplicated so the repeated scalar block appears once and every
    distinct caption is kept."""
    seen: set[str] = set()
    lines: list[str] = []
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        for line in txt.read_text().splitlines():
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return "\n".join(lines)


def run_reason(
    figures_dir: str | Path,
    contracts_dir: str | Path,
    client,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    certainty_threshold: float = DEFAULT_CERTAINTY_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: int | None = None,
) -> dict:
    """Reason over every figure image in `figures_dir` (the ONE token stage).

    Figures are grouped by paper. Once per paper, `reason_paper_scalars` reads
    the paper's text and resolves the conversion scalars (nominal_capacity_mah_g,
    loading_mg_cm2); then each figure is reasoned with `reason_figure`. On an
    "ok" verdict the figure contract is written to `contracts_dir/<stem>.json`,
    with any scalar the figure left null filled from the paper-level value (a
    figure that states its own value wins). skip/flag/error verdicts write no
    contract -- recorded in `reason_manifest.json` with their reason, so the
    figure is dropped rather than guessed. One figure never halts the run.
    """
    figures_dir, contracts_dir = Path(figures_dir), Path(contracts_dir)
    contracts_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in figures_dir.iterdir() if p.suffix.lower() in _MEDIA_TYPES
    )
    if limit is not None:
        images = images[:limit]

    groups: dict[str, list[Path]] = {}
    for img in images:
        groups.setdefault(_paper_of(img.stem), []).append(img)

    ok: list[str] = []
    verdicts: list[dict] = []
    paper_scalars: list[dict] = []
    for paper, imgs in groups.items():
        try:
            ps = reason_paper_scalars(
                client, _paper_text(imgs),
                model=model, fallback_model=fallback_model,
            )
        except Exception as exc:  # noqa: BLE001 -- a bad scalar call never halts the run
            ps = PaperScalars(None, None, f"{type(exc).__name__}: {exc}")
        paper_scalars.append({
            "paper": paper, "nominal_capacity_mah_g": ps.nominal_capacity_mah_g,
            "loading_mg_cm2": ps.loading_mg_cm2, "reason": ps.reason,
        })

        for img in imgs:
            media_type = _MEDIA_TYPES[img.suffix.lower()]
            b64 = base64.standard_b64encode(img.read_bytes()).decode("ascii")
            txt = img.with_suffix(".txt")
            caption = txt.read_text() if txt.exists() else ""
            try:
                res = reason_figure(
                    client, b64, media_type, caption,
                    model=model, fallback_model=fallback_model,
                    max_tokens=max_tokens, certainty_threshold=certainty_threshold,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad call never halts the run
                verdicts.append({"figure": img.stem, "paper": paper, "status": "error",
                                 "reason": f"{type(exc).__name__}: {exc}"})
                continue

            verdict = {"figure": img.stem, "paper": paper, "status": res.status,
                       "reason": res.reason, "certainty": res.certainty}
            if res.status == "ok" and res.contracts:
                # One contract per in-scope panel. A single-panel figure keeps
                # the bare stem; a multi-panel figure writes one contract per
                # panel (<figstem><letter>.json) against a hardlink of the same
                # image (<figstem><letter>.png), so each panel is a standard
                # (image, contract) pair the rest of the pipeline handles.
                written, filled_any = [], set()
                multi = len(res.contracts) > 1
                for label, contract in zip(res.panels, res.contracts):
                    stem = f"{img.stem}{label}" if (multi and label) else img.stem
                    data = contract.model_dump(mode="json")
                    if (data.get("theoretical_capacity_value") is None
                            and ps.nominal_capacity_mah_g is not None):
                        data["theoretical_capacity_value"] = ps.nominal_capacity_mah_g
                        data["theoretical_capacity_basis"] = "specific"
                        filled_any.add("nominal_capacity_mah_g")
                    if data.get("loading_mg_cm2") is None and ps.loading_mg_cm2 is not None:
                        data["loading_mg_cm2"] = ps.loading_mg_cm2
                        filled_any.add("loading_mg_cm2")
                    merged = validate_contract(data)  # re-validate after merge
                    (contracts_dir / f"{stem}.json").write_text(
                        json.dumps(merged.model_dump(mode="json"), indent=2)
                    )
                    if multi:  # give the panel its own image (same bytes)
                        _link_or_copy(img, img.with_name(f"{stem}{img.suffix}"))
                    written.append(stem)
                    ok.append(stem)
                verdict["panels_written"] = written
                verdict["filled_from_paper"] = sorted(filled_any)
            verdicts.append(verdict)

    def _count(status):
        return sum(1 for v in verdicts if v["status"] == status)

    manifest = {
        "model": model,
        "fallback_model": fallback_model,
        "certainty_threshold": certainty_threshold,
        "n_papers": len(groups),
        "n_figures": len(images),
        "n_ok": len(ok),
        "n_skipped": _count("skipped"),
        "n_flagged": _count("flagged"),
        "n_error": _count("error"),
        "ok": ok,
        "paper_scalars": paper_scalars,
        "verdicts": verdicts,
    }
    (contracts_dir / "reason_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_corpus(
    pdf_dir: str | Path,
    out_dir: str | Path,
    client=None,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    certainty_threshold: float = DEFAULT_CERTAINTY_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: int | None = None,
    do_reason: bool = True,
    fresh: bool = False,
    debug: bool = False,
) -> dict:
    """Run all four stages over a directory of PDFs into `out_dir`.

    `client` is required unless `do_reason=False` (which reuses contracts
    already present in `out_dir/contracts`). `fresh=True` reconciles `out_dir`
    to exactly the current PDFs: intake skips re-rendering figures that already
    exist (no redundant identical writes), and every stage subfolder is pruned
    of artifacts whose figure isn't in this run. Returns (and writes as
    `run_manifest.json`) a summary tying the four stage manifests together.
    """
    out = Path(out_dir)
    figures = out / "figures"
    contracts = out / "contracts"
    traces = out / "traces"
    compiled = out / "compiled"
    out.mkdir(parents=True, exist_ok=True)

    intake = intake_run(pdf_dir, figures, skip_existing=fresh)

    pruned: dict[str, list[str]] = {}
    if fresh:
        # Reconcile: keep only what the current PDFs produce, so stale figures
        # aren't re-reasoned (cost) and stale traces aren't re-compiled.
        keep = {f["figstem"] for f in intake.get("figures", [])}
        for name, d in (("figures", figures), ("contracts", contracts),
                        ("traces", traces), ("compiled", compiled)):
            removed = _prune_to(d, keep)
            if removed:
                pruned[name] = removed

    if do_reason:
        if client is None:
            raise ValueError("run_corpus needs an Anthropic client when do_reason=True")
        reason = run_reason(
            figures, contracts, client, model=model,
            fallback_model=fallback_model, certainty_threshold=certainty_threshold,
            max_tokens=max_tokens, limit=limit,
        )
    else:
        contracts.mkdir(parents=True, exist_ok=True)
        reason = {"skipped": "reason stage disabled (--no-reason)"}

    batch = run_batch(str(figures), str(contracts), str(traces), debug=debug)
    compile_manifest = compile_run(str(traces), str(contracts), str(compiled))

    manifest = {
        "pdf_dir": str(pdf_dir),
        "out_dir": str(out),
        "stages": {
            "intake": {"n_papers": intake.get("n_papers"),
                       "n_figures": intake.get("n_figures"),
                       "n_failed": intake.get("n_failed")},
            "reason": {k: reason.get(k) for k in
                       ("n_figures", "n_ok", "n_skipped", "n_flagged", "n_error")}
            if do_reason else reason,
            "batch": {"n_success": batch.get("n_success"),
                      "n_failed": batch.get("n_failed"),
                      "n_no_contract": batch.get("n_no_contract"),
                      "total_findings": batch.get("total_findings")},
            "compile": {"n_compiled": compile_manifest.get("n_compiled"),
                        "n_skipped": compile_manifest.get("n_skipped")},
        },
    }
    if fresh:
        manifest["pruned"] = {k: len(v) for k, v in pruned.items()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_client():
    """Construct the real Anthropic SDK client (credentials from the env or an
    `ant auth login` profile). Imported lazily so intake-only / --no-reason
    runs and the test suite never need the SDK installed or a key set."""
    import anthropic  # noqa: PLC0415 -- lazy on purpose

    return anthropic.Anthropic()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True, help="directory of paper PDFs")
    ap.add_argument("--out-dir", required=True, help="run root (figures/contracts/traces/compiled)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"reasoning model (default {DEFAULT_MODEL})")
    ap.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL,
                    help="refusal fallback model, or 'none' to disable")
    ap.add_argument("--certainty", type=float, default=DEFAULT_CERTAINTY_THRESHOLD,
                    help="min legend<->curve certainty to accept (else flagged)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"reason-call output cap (default {DEFAULT_MAX_TOKENS}); raise "
                         "if a dense figure reports 'output truncated at max_tokens'")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many figures the reason stage processes (cost trial)")
    ap.add_argument("--no-reason", action="store_true",
                    help="skip the token stage; reuse contracts already in out-dir/contracts")
    ap.add_argument("--fresh", action="store_true",
                    help="reconcile out-dir to the current PDFs: prune stale figures/"
                         "contracts/traces/compiled, and skip re-rendering figures that "
                         "already exist (no redundant identical writes)")
    ap.add_argument("--debug", action="store_true", help="save a debug overlay per traced figure")
    args = ap.parse_args()

    fallback = None if (args.fallback_model or "").lower() == "none" else args.fallback_model
    do_reason = not args.no_reason
    client = build_client() if do_reason else None

    manifest = run_corpus(
        args.pdf_dir, args.out_dir, client,
        model=args.model, fallback_model=fallback,
        certainty_threshold=args.certainty, max_tokens=args.max_tokens, limit=args.limit,
        do_reason=do_reason, fresh=args.fresh, debug=args.debug,
    )
    s = manifest["stages"]
    pruned = manifest.get("pruned")
    print(
        f"intake: {s['intake']['n_figures']} figs from {s['intake']['n_papers']} papers | "
        + (f"pruned: {sum(pruned.values())} stale | " if pruned else "")
        + f"reason: {s['reason']} | "
        f"batch: {s['batch']['n_success']} traced | "
        f"compile: {s['compile']['n_compiled']} curves -> {args.out_dir}/compiled"
    )


if __name__ == "__main__":
    main()
