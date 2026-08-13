"""run.py orchestrator tests -- mocked client, no live API calls, no tokens.

One live smoke test at the bottom is gated on ANTHROPIC_API_KEY and skipped by
default, so the normal suite spends nothing.
"""
import base64
import json
import os

import fitz
import pytest

from voltage_tracer.run import build_client, run_corpus, run_reason

# reuse the fake Anthropic client from the reason tests
from voltage_tracer.tests.test_reason import (
    FakeClient, _CONTRACT, _extract, _extract_panels, _panel, _scalars,
)


def _write_png(path, w=40, h=30):
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.get_pixmap().save(str(path))
    doc.close()


def _make_figures(figures_dir, stems):
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _write_png(figures_dir / f"{stem}.png")
        (figures_dir / f"{stem}.txt").write_text(f"caption for {stem}")


# --- run_reason ---


def test_run_reason_writes_contract_only_for_ok(tmp_path):
    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1", "P1_fig2", "P1_fig3"])  # one paper, 3 figures
    contracts = tmp_path / "contracts"
    skip = json.dumps({"decision": "skip", "reason": "multi-cycle", "certainty": 0.9})
    flag = json.dumps({"decision": "uncertain", "reason": "ambiguous", "certainty": 0.5})
    # per-paper scalar call FIRST, then one call per figure
    client = FakeClient([_scalars(), _extract(), skip, flag])

    manifest = run_reason(figures, contracts, client)

    assert manifest["n_papers"] == 1
    assert manifest["n_ok"] == 1 and manifest["n_skipped"] == 1 and manifest["n_flagged"] == 1
    assert (contracts / "P1_fig1.json").exists()
    assert not (contracts / "P1_fig2.json").exists()
    assert not (contracts / "P1_fig3.json").exists()
    assert (contracts / "reason_manifest.json").exists()


def test_run_reason_written_contract_reloads(tmp_path):
    from voltage_tracer.contract import load_contract

    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1"])
    contracts = tmp_path / "contracts"
    run_reason(figures, contracts, FakeClient([_scalars(), _extract()]))
    contract = load_contract(contracts / "P1_fig1.json")  # round-trips through the schema
    assert contract.series[0].c_rate == 0.2
    assert contract.theoretical_capacity_value == 200.0


def test_run_reason_limit_caps_calls(tmp_path):
    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1", "P1_fig2", "P1_fig3"])
    client = FakeClient([_scalars(), _extract(), _extract()])
    manifest = run_reason(figures, tmp_path / "contracts", client, limit=2)
    assert manifest["n_figures"] == 2
    # 1 paper-scalar call + 2 figure calls; the third figure is never reasoned
    assert len(client.calls) == 3


def test_run_reason_records_media_type_from_extension(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    _write_png(figures / "P1_fig1.png")
    (figures / "P1_fig1.txt").write_text("cap")
    client = FakeClient([_scalars(), _extract()])
    run_reason(figures, tmp_path / "contracts", client)
    # calls[0] = paper-scalar (text-only string content); calls[1] = the figure
    assert isinstance(client.calls[0]["messages"][0]["content"], str)
    assert client.calls[1]["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_run_reason_survives_client_exception(tmp_path):
    class Boom:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("network down")
        beta = None

    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1"])
    # non-fable model -> uses .messages.create (which raises on both the
    # paper-scalar call and the figure call); neither halts the run
    manifest = run_reason(figures, tmp_path / "contracts", Boom(), model="claude-opus-4-8")
    assert manifest["n_error"] == 1
    assert "network down" in manifest["verdicts"][0]["reason"]
    # the failed paper-scalar call is recorded too, not swallowed
    assert "network down" in manifest["paper_scalars"][0]["reason"]


def test_paper_scalar_fills_missing_figure_loading(tmp_path):
    from voltage_tracer.contract import load_contract

    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1", "P1_fig2"])
    contracts = tmp_path / "contracts"

    c_missing = {k: v for k, v in _CONTRACT.items() if k != "loading_mg_cm2"}  # no loading
    c_own = {**_CONTRACT, "loading_mg_cm2": 7.0}                                # states its own
    client = FakeClient([
        _scalars(q=180.0, loading=6.4),   # paper-level scalars
        _extract(contract=c_missing),     # P1_fig1: inherits loading
        _extract(contract=c_own),         # P1_fig2: keeps its own
    ])

    manifest = run_reason(figures, contracts, client)

    fig1 = load_contract(contracts / "P1_fig1.json")
    fig2 = load_contract(contracts / "P1_fig2.json")
    assert fig1.loading_mg_cm2 == 6.4          # filled from the paper
    assert fig2.loading_mg_cm2 == 7.0          # figure value wins over the paper's 6.4
    v1 = next(v for v in manifest["verdicts"] if v["figure"] == "P1_fig1")
    v2 = next(v for v in manifest["verdicts"] if v["figure"] == "P1_fig2")
    assert v1["filled_from_paper"] == ["loading_mg_cm2"]
    assert v2["filled_from_paper"] == []
    assert manifest["paper_scalars"][0]["loading_mg_cm2"] == 6.4


def test_run_reason_fills_theoretical_capacity_from_paper(tmp_path):
    from voltage_tracer.contract import Basis, load_contract

    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1"])
    contracts = tmp_path / "contracts"

    # figure states no theoretical capacity; the paper pass supplies it (as a
    # specific nominal capacity).
    drop = ("theoretical_capacity_value", "theoretical_capacity_basis")
    c_missing = {k: v for k, v in _CONTRACT.items() if k not in drop}
    client = FakeClient([_scalars(q=274.0, loading=6.4), _extract(contract=c_missing)])

    manifest = run_reason(figures, contracts, client)
    fig = load_contract(contracts / "P1_fig1.json")
    assert fig.theoretical_capacity_value == 274.0
    assert fig.theoretical_capacity_basis is Basis.specific
    v = next(v for v in manifest["verdicts"] if v["figure"] == "P1_fig1")
    assert "nominal_capacity_mah_g" in v["filled_from_paper"]
    assert manifest["paper_scalars"][0]["nominal_capacity_mah_g"] == 274.0


# --- run_corpus wiring ---


def test_run_corpus_runs_all_stages_and_writes_manifest(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "Body text above the figure.", fontsize=10)
    page.insert_text((72, 600), "Figure 1. Charge/discharge voltage profiles.", fontsize=10)
    doc.save(str(pdfs / "P1.pdf")); doc.close()

    client = FakeClient([_scalars(), _extract()])  # paper-scalar call + the one figure
    out = tmp_path / "run"
    manifest = run_corpus(pdfs, out, client)

    stages = manifest["stages"]
    assert stages["intake"]["n_figures"] == 1
    assert stages["reason"]["n_ok"] == 1
    assert "n_success" in stages["batch"]        # batch stage ran
    assert "n_compiled" in stages["compile"]     # compile stage ran
    assert (out / "run_manifest.json").exists()
    assert (out / "figures").is_dir() and (out / "contracts" / "P1_fig1.json").exists()


def test_run_corpus_no_reason_reuses_existing_contracts(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "Body.", fontsize=10)
    page.insert_text((72, 600), "Figure 1. Voltage profiles.", fontsize=10)
    doc.save(str(pdfs / "P1.pdf")); doc.close()

    out = tmp_path / "run"
    # do_reason=False with no client must not raise and must skip the token stage
    manifest = run_corpus(pdfs, out, client=None, do_reason=False)
    assert "disabled" in manifest["stages"]["reason"]["skipped"]


def test_run_corpus_requires_client_when_reasoning(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    (pdfs / "x.pdf").write_bytes(b"%PDF-1.4\n")  # unreadable but intake tolerates
    with pytest.raises(ValueError, match="needs an Anthropic client"):
        run_corpus(pdfs, tmp_path / "run", client=None, do_reason=True)


def test_build_client_is_lazy_importable():
    # constructs the real SDK client; only asserts the SDK is importable/usable,
    # makes no network call.
    client = build_client()
    assert hasattr(client, "messages")


# --- live smoke test (opt-in; costs tokens) ---


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("VT_LIVE"),
    reason="live API test; set ANTHROPIC_API_KEY and VT_LIVE=1 to run",
)
def test_live_reason_on_real_figure(tmp_path):
    """One real reason call on a real paper figure. Opt-in, costs a few cents."""
    from voltage_tracer.intake import extract_figures
    from voltage_tracer.reason import reason_figure

    papers = "/Users/cbe/Downloads/Nanda Group/SUPER/Solid State NMC Power/Top 21/pdf/Papers/P9.pdf"
    assets = extract_figures(papers, tmp_path / "figs")
    assert assets, "intake found no figures in P9"
    a = assets[0]
    res = reason_figure(build_client(), a.image_b64(), "image/png", a.caption_text())
    assert res.status in {"ok", "skipped", "flagged", "error"}
    if res.status == "ok":
        assert res.contract.series


# --- --fresh reconcile ---


def _one_fig_pdf(path, caption="Figure 1. Voltage profiles."):
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "Body text above the figure.", fontsize=10)
    page.insert_text((72, 600), caption, fontsize=10)
    doc.save(str(path)); doc.close()


def test_fresh_prunes_stale_artifacts_but_keeps_current(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    _one_fig_pdf(pdfs / "P1.pdf")   # produces P1_fig1
    out = tmp_path / "run"
    # seed stale artifacts from a pretend prior run (different paper)
    for sub, names in {
        "figures": ["STALE_fig9.png", "STALE_fig9.txt"],
        "contracts": ["STALE_fig9.json"],
        "traces": ["STALE_fig9__1st_charge.csv", "STALE_fig9__summary.json"],
        "compiled": ["STALE_fig9__1st_charge.csv"],
    }.items():
        d = out / sub; d.mkdir(parents=True)
        for n in names:
            (d / n).write_text("stale")

    manifest = run_corpus(pdfs, out, FakeClient([_scalars(), _extract()]), fresh=True)

    # stale files gone from every subfolder
    assert not (out / "figures" / "STALE_fig9.png").exists()
    assert not (out / "contracts" / "STALE_fig9.json").exists()
    assert not (out / "traces" / "STALE_fig9__1st_charge.csv").exists()
    assert not (out / "compiled" / "STALE_fig9__1st_charge.csv").exists()
    # current run's artifacts present; manifest reports what was pruned
    assert (out / "contracts" / "P1_fig1.json").exists()
    assert manifest["pruned"]["figures"] == 2  # STALE png + txt
    assert manifest["pruned"]["contracts"] == 1


def test_fresh_keeps_manifests_and_current_figure_files(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    _one_fig_pdf(pdfs / "P1.pdf")
    out = tmp_path / "run"
    run_corpus(pdfs, out, FakeClient([_scalars(), _extract()]), fresh=True)
    # a manifest in a pruned dir is never deleted
    assert (out / "figures" / "intake_manifest.json").exists()
    assert (out / "figures" / "P1_fig1.png").exists()
    assert (out / "figures" / "P1_fig1.txt").exists()


def test_fresh_skips_rerender_of_existing_figure(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    _one_fig_pdf(pdfs / "P1.pdf")
    out = tmp_path / "run"
    run_corpus(pdfs, out, FakeClient([_scalars(), _extract()]), fresh=True)
    # clobber the rendered figure, re-run fresh: intake must NOT re-render it
    png = out / "figures" / "P1_fig1.png"
    png.write_bytes(b"SENTINEL")
    run_corpus(pdfs, out, FakeClient([_scalars(), _extract()]), fresh=True)
    assert png.read_bytes() == b"SENTINEL"


# --- multi-panel splitting ---


def test_multi_panel_writes_per_panel_contracts_and_images(tmp_path):
    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig3"])
    contracts = tmp_path / "contracts"
    client = FakeClient([_scalars(), _extract_panels([_panel("a"), _panel("b")])])

    manifest = run_reason(figures, contracts, client)

    # one contract per panel; no bare contract for a multi-panel figure
    assert (contracts / "P1_fig3a.json").exists()
    assert (contracts / "P1_fig3b.json").exists()
    assert not (contracts / "P1_fig3.json").exists()
    # each panel got its own image with the same bytes (hardlink/copy)
    assert (figures / "P1_fig3a.png").read_bytes() == (figures / "P1_fig3.png").read_bytes()
    assert (figures / "P1_fig3b.png").exists()
    v = next(x for x in manifest["verdicts"] if x["figure"] == "P1_fig3")
    assert v["panels_written"] == ["P1_fig3a", "P1_fig3b"]
    assert manifest["n_ok"] == 2  # counts panels


def test_single_panel_keeps_bare_stem(tmp_path):
    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig1"])
    contracts = tmp_path / "contracts"
    run_reason(figures, contracts, FakeClient([_scalars(), _extract_panels([_panel("a")])]))
    assert (contracts / "P1_fig1.json").exists()
    assert not (figures / "P1_fig1a.png").exists()  # no letter suffix for single panel


def test_multi_panel_contracts_are_traceable_pairs(tmp_path):
    # the per-panel (image, contract) pairs must flow through batch like any other
    from voltage_tracer.batch import run_batch

    figures = tmp_path / "figures"
    _make_figures(figures, ["P1_fig3"])
    contracts = tmp_path / "contracts"
    run_reason(figures, contracts, FakeClient([_scalars(), _extract_panels([_panel("a"), _panel("b")])]))
    summary = run_batch(str(figures), str(contracts), str(tmp_path / "traces"))
    # both panels pair with a contract and are traced (P1_fig3.png has no contract)
    traced = {r["figure"] for r in summary["results"] if r["status"] == "success"}
    assert {"P1_fig3a", "P1_fig3b"} <= traced
