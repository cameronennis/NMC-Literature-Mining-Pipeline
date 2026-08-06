"""Intake stage: PDF -> candidate figure images + their caption/methods text.

Deterministic and token-free. Produces exactly the two inputs `reason.py`
needs per figure -- a cropped figure image and the text that describes it --
named on the same `<figstem>` convention the rest of the pipeline pairs on
(`<figstem>.png` <-> `<figstem>.json` <-> `<figstem>__<series_id>.csv`).

How a figure is located
-----------------------
Captions are found by regex on the page's text blocks ("Figure 3.", "Fig. 3"),
then the figure is taken to be the region ABOVE its caption, bounded
horizontally by the caption's own extent and vertically by the nearest text
block above it. This is a heuristic, not layout understanding -- it is right
for the overwhelmingly common single-column/one-figure-per-caption case in
this corpus, and when it is wrong it errs toward including too much (extra
whitespace or a neighbouring panel), which the contract stage tolerates: the
model reports the axes rectangle as `plot_bbox_px` and the pipeline crops to
it.

Caption prefilter
-----------------
At 1000 papers the dominant cost is paying vision tokens for figures that the
contract stage then declines. Captions are prefiltered on voltage-profile
keywords first, so most irrelevant figures are dropped for free. This is a
recall/cost trade: widen `CAPTION_KEYWORDS` (or pass `keywords=()`) to render
everything.

Known limitation
----------------
A multi-panel figure yields ONE image, so if two panels are both first-cycle
voltage profiles only one contract can describe it. The contract stage flags
this rather than guessing.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# Vision models gain nothing above this and every extra pixel costs tokens.
MAX_EDGE_PX = 2576
# Don't render a "figure" region smaller than this (points) -- it's a stray
# caption-like line, not a plot.
MIN_REGION_PT = 60.0
# Page margin used when no text block sits above the caption.
PAGE_MARGIN_PT = 36.0

_CAPTION_RE = re.compile(r"^\s*(?:Fig(?:ure)?\.?)\s*([0-9]+|[SI]+\s*[0-9]+)", re.IGNORECASE)
# Panel letters that are properly terminated ("4a-e." / "4a:") are part of a
# caption; bare ones ("5a-e show ...") mean an inline cross-reference.
_PANEL_PREFIX_RE = re.compile(r"^[a-z](?:\s*[-–]\s*[a-z])?\s*[.:]")

# Captions worth spending vision tokens on. Deliberately broad -- the contract
# stage does the real first-cycle scope judgement.
CAPTION_KEYWORDS = (
    "voltage", "charge", "discharge", "profile", "galvanostatic",
    "capacity", "cycl", "curve", "polarization", "potential",
)

# Sentence boundary: a period that is NOT between two digits, so decimal
# values survive intact ("36.94 mg/cm2" must not be cut to "36."). Those
# decimals ARE the scalars we're harvesting, so this is load-bearing.
_SENTENCE_SPLIT = re.compile(r"(?<!\d)[.;](?=\s|$)")

# Markers of a sentence carrying the conversion scalars compile.py needs.
_SCALAR_MARKERS = (
    re.compile(r"\bloading\b", re.IGNORECASE),
    re.compile(r"mg\s*[·\s]*cm\s*[-−–]?\s*2", re.IGNORECASE),
    re.compile(r"mA\s*h\s*[·\s]*g\s*[-−–]?\s*1", re.IGNORECASE),
    re.compile(r"\bmAh\s*/?\s*g\b", re.IGNORECASE),
    re.compile(r"\b1\s*C\b"),
)
MAX_SCALAR_SENTENCES = 12
MAX_SENTENCE_CHARS = 400


@dataclass
class FigureAsset:
    """One candidate figure ready for the reasoning stage."""

    figstem: str
    image_path: Path
    text_path: Path
    page_number: int  # 1-based, as printed in a PDF reader
    caption: str

    def image_b64(self) -> str:
        return base64.standard_b64encode(self.image_path.read_bytes()).decode("ascii")

    def caption_text(self) -> str:
        return self.text_path.read_text()


@dataclass
class _Caption:
    number: str
    text: str
    rect: tuple[float, float, float, float]


def _is_caption(text: str) -> re.Match | None:
    """Match a real caption, rejecting inline cross-references.

    Body text routinely opens a sentence with "Fig. 5a-e show CV tests ...".
    A caption continues with a capital letter or a panel parenthesis
    ("Fig. 3. (a) Charge-discharge curves"), whereas a cross-reference
    continues with a lowercase verb -- that's the discriminator.
    """
    stripped = text.strip()
    m = _CAPTION_RE.match(stripped)
    if not m:
        return None
    rest = stripped[m.end():].lstrip(" .:")
    panel = _PANEL_PREFIX_RE.match(rest)
    if panel:  # "4a-e." -> a genuine panel-lettered caption; skip past it
        rest = rest[panel.end():].lstrip(" .:")
    if rest[:1].islower():
        return None
    return m


def _caption_is_relevant(text: str, keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return True
    low = text.lower()
    return any(k in low for k in keywords)


def _find_captions(page) -> list[_Caption]:
    """Text blocks on the page that begin like a figure caption."""
    out = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        m = _is_caption(text)
        if m:
            out.append(
                _Caption(
                    number=re.sub(r"\s+", "", m.group(1)),
                    text=" ".join(text.split()),
                    rect=(x0, y0, x1, y1),
                )
            )
    return out


def _figure_rect(page, caption: _Caption) -> tuple[float, float, float, float] | None:
    """Region above `caption` bounded by the nearest text block above it."""
    cx0, cy0, cx1, _cy1 = caption.rect
    top = PAGE_MARGIN_PT
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4]
        if not text.strip():
            continue
        if by1 >= cy0:  # not above the caption
            continue
        # only blocks that horizontally overlap the caption's column
        if bx1 < cx0 or bx0 > cx1:
            continue
        top = max(top, by1)
    if cy0 - top < MIN_REGION_PT or cx1 - cx0 < MIN_REGION_PT:
        return None
    return (cx0, top, cx1, cy0)


def _render(page, rect, out_path: Path) -> None:
    """Render `rect` (PDF points) to PNG, capped at MAX_EDGE_PX on the long edge."""
    x0, y0, x1, y1 = rect
    long_edge_pt = max(x1 - x0, y1 - y0)
    # 300 dpi baseline (72 pt/inch), then clamp so we never exceed the cap.
    zoom = min(300.0 / 72.0, MAX_EDGE_PX / long_edge_pt)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(x0, y0, x1, y1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))


def extract_scalar_sentences(doc) -> list[str]:
    """Sentences from anywhere in the document that may carry q_1c_mah_g /
    loading_mg_cm2. Passed to the reasoning stage alongside the caption so it
    can fill the conversion scalars without a second pass over the paper."""
    full = " ".join(page.get_text() for page in doc)
    full = " ".join(full.split())
    found: list[str] = []
    seen: set[str] = set()
    for raw in _SENTENCE_SPLIT.split(full):
        s = raw.strip()
        if not s or len(s) > MAX_SENTENCE_CHARS:
            continue
        if not any(p.search(s) for p in _SCALAR_MARKERS):
            continue
        if s in seen:
            continue
        seen.add(s)
        found.append(s)
        if len(found) >= MAX_SCALAR_SENTENCES:
            break
    return found


def extract_figures(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    keywords: tuple[str, ...] = CAPTION_KEYWORDS,
    skip_existing: bool = False,
) -> list[FigureAsset]:
    """Render every relevant figure in `pdf_path` into `out_dir`.

    Writes `<pdfstem>_fig<N>.png` and `<pdfstem>_fig<N>.txt` per figure, where
    the .txt holds the caption plus any scalar-bearing sentences from the
    paper. Returns the assets in page order.

    `skip_existing=True` skips rendering (and rewriting the .txt for) any
    figure whose .png and .txt already exist -- the same PDF region renders to
    the same bytes, so re-doing it just churns identical files. Used by the
    `--fresh` reconcile path. To force a re-render, delete the files first.
    """
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: list[FigureAsset] = []

    with fitz.open(pdf_path) as doc:
        scalars = extract_scalar_sentences(doc)
        scalar_block = (
            "\n\nPossibly relevant sentences from the paper (for q_1c_mah_g / "
            "loading_mg_cm2):\n" + "\n".join(f"- {s}" for s in scalars)
            if scalars else ""
        )
        used: set[str] = set()
        for page_index, page in enumerate(doc):
            for caption in _find_captions(page):
                if not _caption_is_relevant(caption.text, keywords):
                    continue
                rect = _figure_rect(page, caption)
                if rect is None:
                    continue
                figstem = f"{pdf_path.stem}_fig{caption.number}"
                # Two captions parsing to the same number (e.g. a continued
                # figure) must not overwrite each other.
                suffix = 1
                base = figstem
                while figstem in used:
                    suffix += 1
                    figstem = f"{base}_{suffix}"
                used.add(figstem)

                image_path = out_dir / f"{figstem}.png"
                text_path = out_dir / f"{figstem}.txt"
                if not (skip_existing and image_path.exists() and text_path.exists()):
                    _render(page, rect, image_path)
                    text_path.write_text(caption.text + scalar_block)
                assets.append(
                    FigureAsset(
                        figstem=figstem,
                        image_path=image_path,
                        text_path=text_path,
                        page_number=page_index + 1,
                        caption=caption.text,
                    )
                )
    return assets


def intake_run(pdf_dir: str | Path, out_dir: str | Path, *,
               skip_existing: bool = False) -> dict:
    """Run `extract_figures` over every PDF in `pdf_dir`.

    One unreadable PDF never halts the run -- it is recorded and skipped.
    `skip_existing` is forwarded to `extract_figures` (see its docstring).
    Writes (and returns) `intake_manifest.json`.
    """
    pdf_dir, out_dir = Path(pdf_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []
    failed: list[dict] = []

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        try:
            for a in extract_figures(pdf, out_dir, skip_existing=skip_existing):
                figures.append({
                    "figstem": a.figstem,
                    "paper": pdf.stem,
                    "page": a.page_number,
                    "caption": a.caption,
                })
        except Exception as exc:  # noqa: BLE001 -- keep the corpus run alive
            failed.append({"paper": pdf.stem, "error": f"{type(exc).__name__}: {exc}"})

    manifest = {
        "n_papers": len(list(pdf_dir.glob("*.pdf"))),
        "n_figures": len(figures),
        "n_failed": len(failed),
        "figures": figures,
        "failed": failed,
    }
    (out_dir / "intake_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
