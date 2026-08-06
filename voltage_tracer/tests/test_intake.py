"""intake.py tests -- synthetic PDFs built with PyMuPDF, fully deterministic."""
import json

import fitz
import pytest

from voltage_tracer.intake import (
    MAX_EDGE_PX,
    extract_figures,
    extract_scalar_sentences,
    intake_run,
)


def _make_pdf(path, pages):
    """Build a PDF. `pages` is a list of lists of (x, y, text) draw commands;
    a rectangle is also drawn on each page so figure regions have ink."""
    doc = fitz.open()
    for items in pages:
        page = doc.new_page(width=595, height=842)  # A4 points
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=10)
        doc.save  # no-op, keeps intent explicit
    doc.save(str(path))
    doc.close()


def _fig_page(caption, body_y=120, caption_y=600):
    """A page with a heading, a big empty figure area, then a caption."""
    return [
        (72, body_y, "Some body text well above the figure."),
        (72, caption_y, caption),
    ]


def test_extracts_relevant_figure_with_image_and_text(tmp_path):
    pdf = tmp_path / "P9.pdf"
    _make_pdf(pdf, [_fig_page("Figure 6. Charge/discharge voltage profiles of the cell.")])
    assets = extract_figures(pdf, tmp_path / "out")
    assert len(assets) == 1
    a = assets[0]
    assert a.figstem == "P9_fig6"
    assert a.page_number == 1
    assert a.image_path.exists() and a.image_path.suffix == ".png"
    assert a.text_path.exists()
    assert "voltage profiles" in a.caption_text()
    assert a.image_b64()  # non-empty base64


def test_irrelevant_caption_is_filtered_before_rendering(tmp_path):
    pdf = tmp_path / "P1.pdf"
    _make_pdf(pdf, [_fig_page("Figure 2. XRD patterns of the pristine powder.")])
    out = tmp_path / "out"
    assert extract_figures(pdf, out) == []
    # nothing rendered -> no vision tokens would be spent on it
    assert not list(out.glob("*.png"))


def test_keyword_filter_can_be_disabled(tmp_path):
    pdf = tmp_path / "P1.pdf"
    _make_pdf(pdf, [_fig_page("Figure 2. XRD patterns of the pristine powder.")])
    assets = extract_figures(pdf, tmp_path / "out", keywords=())
    assert len(assets) == 1


def test_multiple_pages_are_extracted_in_order(tmp_path):
    pdf = tmp_path / "P7.pdf"
    _make_pdf(pdf, [
        _fig_page("Figure 1. Discharge capacity retention."),
        _fig_page("Figure 4. Voltage profiles at various C-rates."),
    ])
    assets = extract_figures(pdf, tmp_path / "out")
    assert [a.figstem for a in assets] == ["P7_fig1", "P7_fig4"]
    assert [a.page_number for a in assets] == [1, 2]


def test_duplicate_figure_numbers_do_not_overwrite(tmp_path):
    pdf = tmp_path / "P3.pdf"
    cap = "Figure 5. Voltage profile continued."
    _make_pdf(pdf, [_fig_page(cap), _fig_page(cap)])
    assets = extract_figures(pdf, tmp_path / "out")
    stems = [a.figstem for a in assets]
    assert stems == ["P3_fig5", "P3_fig5_2"]
    assert len(set(p.image_path for p in assets)) == 2


def test_fig_abbreviation_and_si_numbering_are_recognised(tmp_path):
    pdf = tmp_path / "P10_SI.pdf"
    _make_pdf(pdf, [_fig_page("Fig. S3 Galvanostatic charge curves.")])
    assets = extract_figures(pdf, tmp_path / "out")
    assert len(assets) == 1
    assert assets[0].figstem == "P10_SI_figS3"


def test_caption_with_no_room_above_is_skipped(tmp_path):
    # caption near the very top -> region below MIN_REGION_PT
    pdf = tmp_path / "P2.pdf"
    _make_pdf(pdf, [[(72, 50, "Figure 1. Voltage profiles.")]])
    assert extract_figures(pdf, tmp_path / "out") == []


def test_rendered_image_respects_resolution_cap(tmp_path):
    pdf = tmp_path / "P4.pdf"
    _make_pdf(pdf, [_fig_page("Figure 1. Voltage profiles.", body_y=60, caption_y=800)])
    assets = extract_figures(pdf, tmp_path / "out")
    with fitz.open(assets[0].image_path) as img:
        page = img[0]
        assert max(page.rect.width, page.rect.height) <= MAX_EDGE_PX


def test_scalar_sentences_are_harvested_into_text(tmp_path):
    pdf = tmp_path / "P9.pdf"
    _make_pdf(pdf, [
        _fig_page("Figure 6. Voltage profiles of the cell."),
        [(72, 100, "The cathode loading was 36.94 mg/cm2 in all cells."),
         (72, 130, "A theoretical capacity of 200 mAh/g was assumed for 1 C.")],
    ])
    assets = extract_figures(pdf, tmp_path / "out")
    text = assets[0].caption_text()
    assert "36.94" in text
    assert "200" in text


def test_extract_scalar_sentences_caps_output(tmp_path):
    pdf = tmp_path / "big.pdf"
    lines = [(72, 100 + 15 * i, f"The loading was {i} mg/cm2 here.") for i in range(30)]
    _make_pdf(pdf, [lines])
    with fitz.open(pdf) as doc:
        assert len(extract_scalar_sentences(doc)) <= 12


def test_intake_run_writes_manifest_and_survives_bad_pdf(tmp_path):
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    _make_pdf(pdfs / "P9.pdf", [_fig_page("Figure 6. Voltage profiles.")])
    (pdfs / "broken.pdf").write_text("not a pdf at all")
    out = tmp_path / "out"
    manifest = intake_run(pdfs, out)
    assert manifest["n_papers"] == 2
    assert manifest["n_figures"] == 1
    assert manifest["n_failed"] == 1
    assert manifest["failed"][0]["paper"] == "broken"
    assert (out / "intake_manifest.json").exists()
    saved = json.loads((out / "intake_manifest.json").read_text())
    assert saved["figures"][0]["figstem"] == "P9_fig6"


def test_figstem_matches_pipeline_pairing_convention(tmp_path):
    # intake's .png stem must be what batch.py/compile.py pair contracts on
    pdf = tmp_path / "P9.pdf"
    _make_pdf(pdf, [_fig_page("Figure 6. Voltage profiles.")])
    a = extract_figures(pdf, tmp_path / "out")[0]
    assert a.image_path.stem == a.figstem
    assert a.text_path.stem == a.figstem


def test_inline_cross_reference_is_not_a_caption(tmp_path):
    # Body text opening "Fig. 5a-e show ..." is a reference, not a caption.
    pdf = tmp_path / "P7.pdf"
    _make_pdf(pdf, [_fig_page("Fig. 5a-e show CV curves at different scan rates.")])
    assert extract_figures(pdf, tmp_path / "out") == []


def test_panel_lettered_caption_is_still_accepted(tmp_path):
    pdf = tmp_path / "P7.pdf"
    _make_pdf(pdf, [_fig_page("Fig. 4a-e. Charge/discharge profiles of the cell.")])
    assets = extract_figures(pdf, tmp_path / "out")
    assert [a.figstem for a in assets] == ["P7_fig4"]


def test_caption_with_parenthesised_panels_is_accepted(tmp_path):
    pdf = tmp_path / "P7.pdf"
    _make_pdf(pdf, [_fig_page("Fig. 3. (a) Charge-discharge curves of SC-Ni92.")])
    assert [a.figstem for a in extract_figures(pdf, tmp_path / "out")] == ["P7_fig3"]


def test_skip_existing_does_not_rewrite_identical_figure(tmp_path):
    pdf = tmp_path / "P9.pdf"
    _make_pdf(pdf, [_fig_page("Figure 6. Voltage profiles.")])
    out = tmp_path / "out"
    # first pass renders it
    a1 = extract_figures(pdf, out)[0]
    # clobber both sidecars with sentinels; skip_existing must leave them
    a1.image_path.write_bytes(b"SENTINEL_PNG")
    a1.text_path.write_text("SENTINEL_TXT")
    a2 = extract_figures(pdf, out, skip_existing=True)
    assert len(a2) == 1 and a2[0].figstem == a1.figstem
    assert a1.image_path.read_bytes() == b"SENTINEL_PNG"   # not re-rendered
    assert a1.text_path.read_text() == "SENTINEL_TXT"      # not rewritten


def test_skip_existing_still_renders_a_missing_figure(tmp_path):
    pdf = tmp_path / "P9.pdf"
    _make_pdf(pdf, [_fig_page("Figure 6. Voltage profiles.")])
    out = tmp_path / "out"
    a = extract_figures(pdf, out, skip_existing=True)  # nothing there yet
    assert len(a) == 1 and a[0].image_path.exists() and a[0].text_path.exists()
