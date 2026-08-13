"""Reasoning stage: figure image + caption -> validated Contract.

This is the ONE operation in the whole pipeline that costs tokens. Everything
downstream (trace, smooth, calibrate, validate, compile) is deterministic
Python. For a single figure this module makes exactly one Anthropic API call
and returns a validated `Contract` (or a skip/flag/error verdict) -- the
image-reading judgement the deterministic code can't do itself.

Design rules honoured here:
  * The API client is INJECTED (`reason_figure(client, ...)`), so tests supply
    a mock and stay deterministic and token-free.
  * The fixed instruction block is a cached system prompt; the per-figure
    image + caption are the only volatile bytes, so repeated calls reuse the
    prompt cache.
  * Validate, don't trust: the model's JSON is parsed and fed through the
    same `Contract` schema the rest of the pipeline uses. On a parse/validation
    failure we retry once with a corrective note, then flag-and-skip -- a bad
    contract is never written.
  * Scope + certainty gates (first-cycle-only; >=95% legend<->curve certainty)
    are asked of the model and enforced here, so out-of-scope or low-confidence
    figures become `skipped` / `flagged` verdicts, never silent bad data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from voltage_tracer.contract import Contract, validate_contract

# ---------------------------------------------------------------------------
# MODEL CONFIG -- the single place the model is chosen. `run.py` also exposes
# this as a --model flag, and `reason_figure(..., model=...)` overrides per
# figure, so you should rarely need to edit source.
#
#   "claude-fable-5"   the default: strongest vision/reasoning. Reading tick
#                      pixels, curve colours and legend<->curve mapping off
#                      dense figures is the accuracy bottleneck for the whole
#                      project, so it leads here. ~2x Opus cost.
#   "claude-opus-4-8"  ~half the cost, very close on most figures.
#   "claude-sonnet-5"  cheapest; fine for clean single-panel figures.
DEFAULT_MODEL = "claude-fable-5"
# Refusal fallback. Fable/Mythos run request-level safety classifiers that can
# decline a request (`stop_reason == "refusal"`) -- occasionally a false
# positive on benign battery text. With this set, the API automatically re-runs
# the refused request on this model IN THE SAME CALL, so one bad classifier hit
# doesn't cost the figure. No-op for non-Fable models (see `_fallbacks_for`).
# Set to None to disable.
DEFAULT_FALLBACK_MODEL = "claude-opus-4-8"
FALLBACK_BETA = "server-side-fallback-2026-06-01"
# A dense multi-panel rate-capability figure (e.g. 3 panels x ~16 curves) emits
# a large JSON -- one series object per curve, per panel. Too small a cap
# truncates the JSON mid-string and the figure fails to parse. 8000 comfortably
# covers the densest figures in the corpus; raise --max-tokens if a figure
# still reports "output truncated at max_tokens".
DEFAULT_MAX_TOKENS = 8000
DEFAULT_CERTAINTY_THRESHOLD = 0.95

# --- Fixed instruction block (BYTE-STABLE so the prompt cache hits). --------
# Do not interpolate anything volatile (dates, ids, the caption) into this.
INSTRUCTIONS = """You read one figure from a solid-state NMC battery paper and \
emit JSON describing its voltage-vs-capacity charge/discharge curves, for a \
deterministic graph-tracing pipeline.

SCOPE -- which curves to describe. Describe voltage-vs-capacity charge/discharge \
profiles that can be compared as power vs. time. IN scope:
  * first-cycle charge/discharge profiles;
  * RATE-CAPABILITY profiles -- one charge/discharge curve per C-rate (e.g.
    0.1C / 0.2C / ... / 20C). These ARE in scope even though the higher-rate
    curves are recorded at later cycles; they differ by C-RATE, not by fade.
DECLINE only curves whose purpose is to show CYCLE DECAY / degradation: the same \
cell at the same rate drawn at several cycle numbers (cycle 1 vs 50 vs 100 ...), \
"cycle number" on an axis, or capacity-retention / cycling-performance plots. \
The test: if the curves differ by C-RATE or by CELL/coating, KEEP them; if they \
differ by CYCLE NUMBER to show fade, DECLINE them.

MULTI-PANEL FIGURES. A figure often holds several panels (a, b, c, ...). Treat \
EACH in-scope voltage-profile panel separately -- one entry per panel in \
"panels" below, each with its own plot box, axes and curves. OMIT panels that \
are out of scope (rate-capability bar charts, CV, cycling/retention, Nyquist, \
SEM/TEM, XRD, schematics, etc.); include only charge/discharge voltage-profile \
panels. A single-panel figure yields a "panels" list of length 1.

Return ONE JSON object, no prose, no markdown fences:
{
  "decision": "extract" | "skip" | "uncertain",
  "reason": "<one short sentence>",
  "certainty": <number 0..1>,          // confidence in the legend<->curve
                                        // (colour/label/direction/rate) mapping
  "loading_mg_cm2": <number or null>,  // figure-level: electrode areal loading
                                        // (mg/cm2), from the caption/text
  "theoretical_capacity_value": <number or null>,  // nominal capacity a C-rate is
  "theoretical_capacity_basis": "specific" | "areal" | null,  // defined against,
                                        // e.g. "1C = 200 mAh/g" -> 200,"specific"
  "panels": [ <panel object>, ... ]    // one per in-scope panel; [] if none
}

Use "skip" if the figure has NO in-scope voltage-profile panel (only cycle-decay, \
CV, rate bars, microscopy, etc.). Use "uncertain" (or certainty below 0.95) when \
you cannot confidently map each curve to its colour, direction and rate. Only \
use "extract" when confident, with one entry per in-scope panel.

Each panel object:
  "panel": "<panel letter, e.g. a>",
  "plot_bbox_px": [x0,y0,x1,y1],   // THIS panel's axes rectangle, in the FULL
                                    // image's pixels (top-left origin; x1>x0,
                                    // y1>y0)
  "x_capacity_basis": "specific" | "areal",  // what the x-axis measures: read the
                                    // AXIS LABEL -- "specific"=per gram (mAh/g),
                                    // "areal"=per area (mAh/cm2). Default specific.
  "x_axis" / "y_axis": {
      "scale": "linear" | "log",
      "ticks_px":  [pixel positions of >=2 labelled ticks],
      "ticks_val": [their numeric values, same length]
  },  // x is capacity, y is voltage (V); y ticks_px top-to-bottom so each pixel
      // pairs with its voltage. >=2 DISTINCT ticks per axis.
  "series": [ one entry per curve in THIS panel:
     {
       "id": "<stable slug, e.g. 0.2C_discharge>",
       "color_rgb": [r,g,b],          // the curve's colour, 0..255
       "label": "<legend text>",
       "direction": "increasing" | "decreasing",  // charge rises, discharge falls
       // How this curve's current is set -- give EITHER a c_rate OR a
       // current_value+current_basis (prefer c_rate if the figure labels rates):
       "c_rate": <number or null>,    // e.g. 0.2 for 0.2C; null if unlabelled
       "current_value": <number or null>,       // a stated current for this curve
       "current_basis": "specific" | "areal" | null,  // mA/g vs mA/cm2
       "loading_mg_cm2": <number or null>,  // per-curve loading, ONLY when curves
                                        // differ by mass loading (else null ->
                                        // inherit the figure-level loading)
       "c_rate_rank": <int or null>   // optional ordering hint, slowest=low
     } ],
  "legend_bbox_px": [x0,y0,x1,y1] or null,   // legend rectangle to mask, if any
  "data_region_frac": [fx0,fy0,fx1,fy1]      // sub-rectangle of THIS panel's plot
                                             // box that curve data occupies, as
                                             // fractions (default [0,0,1,1]);
                                             // shrink to exclude in-plot titles,
                                             // panel letters, or an over-axes
                                             // legend.

CONVERSION NOTES. The pipeline turns each curve into areal power vs. time; it needs \
an areal current per curve and an areal capacity axis. Fill what the figure/text \
states, leave the rest null -- never guess:
  * loading_mg_cm2 (figure-level, and per-curve when curves differ by loading) is \
the bridge between per-gram and per-area; report it whenever stated.
  * a C-rate needs a nominal capacity to become a current. Report \
theoretical_capacity_value+basis when the paper states one (e.g. "theoretical \
capacity 274 mAh/g", "1C = 200 mAh/g"). If it doesn't, leave them null -- the \
pipeline will fall back to the slowest curve. So a rate-capability panel that shows \
a slow (<=1C) curve is fine even with no stated capacity.
All pixel coordinates are in the FULL image's pixel space."""


@dataclass(frozen=True)
class ReasonResult:
    """Outcome of reasoning over one figure.

    status:
      "ok"      -> `contracts` holds one validated Contract per in-scope panel
                   (parallel to `panels`, the panel letters), ready for the
                   pipeline. A single-panel figure yields one contract.
      "skipped" -> out of scope (no in-scope voltage-profile panel).
      "flagged" -> in scope but low certainty; needs human review.
      "error"   -> the model's output could not be parsed/validated after a
                   retry, so the figure is dropped (never written as a guess).
    """

    status: str
    contracts: list[Contract]
    panels: list[str]  # panel letters, parallel to contracts ("" for single-panel)
    reason: str
    certainty: float | None = None
    raw: str | None = None  # raw model text, kept for debugging failures

    @property
    def contract(self) -> Contract | None:
        """The first contract, or None. Back-compat for the single-panel path."""
        return self.contracts[0] if self.contracts else None


@dataclass(frozen=True)
class PaperScalars:
    """Paper-level conversion scalars, resolved once per paper.

    nominal_capacity_mah_g (the specific capacity a C-rate is defined against)
    and loading_mg_cm2 are properties of the cathode/electrode, not of any
    single figure, so they are extracted once from the paper's text and used to
    fill any figure contract that left them null (a figure that states its own
    value still wins). Either may be None if the paper doesn't state it.
    """

    nominal_capacity_mah_g: float | None
    loading_mg_cm2: float | None
    reason: str = ""


def _fallbacks_for(model: str, fallback_model: str | None) -> str | None:
    """The model to fall back to on a policy refusal, or None.

    Only Fable/Mythos run the request-level safety classifiers that produce a
    `refusal` stop reason, and only they accept the server-side `fallbacks`
    parameter -- so this is scoped to those models. Switching DEFAULT_MODEL
    back to Opus/Sonnet silently drops the fallback and the plain (non-beta)
    endpoint is used again.
    """
    if fallback_model is None:
        return None
    if model.startswith(("claude-fable", "claude-mythos")):
        return fallback_model
    return None


def _create(client, model, fallback_model, **kwargs):
    """One API call, on the beta endpoint only when a fallback is in play."""
    fb = _fallbacks_for(model, fallback_model)
    if fb is None:
        return client.messages.create(model=model, **kwargs)
    return client.beta.messages.create(
        model=model, betas=[FALLBACK_BETA], fallbacks=[{"model": fb}], **kwargs
    )


def _refusal_reason(response) -> str | None:
    """If the request was declined by safety classifiers, describe it.

    A refusal is a SUCCESSFUL HTTP 200 with `stop_reason == "refusal"` and an
    empty (or partial) `content` -- so this must be checked before reading any
    content block. With a fallback configured, reaching here means the whole
    chain refused.
    """
    if getattr(response, "stop_reason", None) != "refusal":
        return None
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) if details is not None else None
    return f"model declined the request (category: {category or 'unspecified'})"


def _extract_text(response) -> str:
    """Concatenate the text blocks of an Anthropic message response."""
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _strip_fences(text: str) -> str:
    """Tolerate a ```json ... ``` wrapper if the model adds one."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _build_user_content(image_b64: str, media_type: str, caption_text: str) -> list:
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
        },
        {
            "type": "text",
            "text": (
                "Caption / methods text for this figure (may be empty):\n"
                f"{caption_text or '(none provided)'}\n\n"
                "Return the JSON object described in the instructions."
            ),
        },
    ]


def _parse_envelope(text: str) -> dict:
    """Parse the model's JSON envelope; raises ValueError on malformed JSON."""
    try:
        obj = json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("response JSON was not an object")
    return obj


def reason_figure(
    client,
    image_b64: str,
    media_type: str,
    caption_text: str,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    certainty_threshold: float = DEFAULT_CERTAINTY_THRESHOLD,
) -> ReasonResult:
    """Reason over one figure and return a validated Contract or a verdict.

    `client` is any object exposing `messages.create(...)` (the Anthropic SDK
    client, or a mock in tests). `image_b64` is the base64-encoded figure image
    and `media_type` its MIME type (e.g. "image/png"). Makes at most two API
    calls: the initial attempt, plus one corrective retry if the first output
    fails to parse or validate.

    No `thinking` parameter is sent: Fable/Mythos have thinking always on and
    reject explicit configuration, while Opus/Sonnet run fine without it.
    """
    system = [{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": _build_user_content(image_b64, media_type, caption_text)}]

    last_error = ""
    last_raw = ""
    for attempt in range(2):  # one initial try + one corrective retry
        response = _create(
            client, model, fallback_model,
            max_tokens=max_tokens, system=system, messages=messages,
        )
        # Must precede any content access -- a refusal can have empty content.
        refusal = _refusal_reason(response)
        if refusal is not None:
            return ReasonResult("flagged", [], [], refusal, None, None)
        raw = _extract_text(response)
        last_raw = raw
        truncated = getattr(response, "stop_reason", None) == "max_tokens"
        try:
            envelope = _parse_envelope(raw)
            return _finish(envelope, certainty_threshold, raw)
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            if truncated:
                # A retry with the same cap truncates again -- report clearly
                # instead of burning the retry.
                return ReasonResult(
                    "error", [], [],
                    f"output truncated at max_tokens={max_tokens}; the figure is "
                    f"dense (many panels/curves) -- raise --max-tokens",
                    raw=raw,
                )
            # Feed the failure back for a single corrective retry.
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That could not be used: {last_error}. "
                        "Reply again with ONLY the JSON object described in the "
                        "instructions, correcting the problem."
                    ),
                },
            ]

    return ReasonResult(
        status="error",
        contracts=[],
        panels=[],
        reason=f"model output invalid after retry: {last_error}",
        raw=last_raw,
    )


def _panel_labels(raw: list[str]) -> list[str]:
    """Clean, de-duplicated panel letters; fall back to a,b,c... by index."""
    import string

    out: list[str] = []
    seen: set[str] = set()
    for i, lab in enumerate(raw):
        lab = "".join(ch for ch in lab.lower() if ch.isalnum())
        if not lab or lab in seen:
            lab = string.ascii_lowercase[i] if i < 26 else f"p{i}"
        seen.add(lab)
        out.append(lab)
    return out


def _finish(envelope: dict, certainty_threshold: float, raw: str) -> ReasonResult:
    """Turn a parsed envelope into a ReasonResult, enforcing the gates.

    Raises ValueError/ValidationError on malformed extract envelopes so the
    caller's retry loop can catch them.
    """
    decision = envelope.get("decision")
    reason = str(envelope.get("reason", "") or "")
    certainty = envelope.get("certainty")
    certainty = float(certainty) if isinstance(certainty, (int, float)) else None

    if decision == "skip":
        return ReasonResult("skipped", [], [], reason or "out of scope", certainty, raw)

    if decision == "uncertain":
        return ReasonResult("flagged", [], [], reason or "model uncertain", certainty, raw)

    if decision != "extract":
        raise ValueError(f"unknown decision {decision!r} (expected extract/skip/uncertain)")

    if certainty is not None and certainty < certainty_threshold:
        return ReasonResult(
            "flagged", [], [],
            reason or f"certainty {certainty} below {certainty_threshold}",
            certainty, raw,
        )

    # One panel object per in-scope panel; tolerate a legacy single "contract".
    panels_in = envelope.get("panels")
    if panels_in is None and isinstance(envelope.get("contract"), dict):
        panels_in = [envelope["contract"]]
    if not isinstance(panels_in, list) or not panels_in:
        raise ValueError("decision 'extract' requires a non-empty 'panels' list")

    fig_load = envelope.get("loading_mg_cm2")
    fig_theo_val = envelope.get("theoretical_capacity_value")
    fig_theo_basis = envelope.get("theoretical_capacity_basis")

    contracts: list[Contract] = []
    raw_labels: list[str] = []
    for p in panels_in:
        if not isinstance(p, dict):
            raise ValueError("each panel must be a JSON object")
        raw_labels.append(str(p.get("panel", "") or ""))
        data = {k: v for k, v in p.items() if k != "panel"}
        # figure-level conversion metadata defaults into each panel unless the
        # panel states its own (panel value wins).
        if data.get("loading_mg_cm2") is None and fig_load is not None:
            data["loading_mg_cm2"] = fig_load
        if data.get("theoretical_capacity_value") is None and fig_theo_val is not None:
            data["theoretical_capacity_value"] = fig_theo_val
            if data.get("theoretical_capacity_basis") is None:
                data["theoretical_capacity_basis"] = fig_theo_basis
        contracts.append(validate_contract(data))  # raises ValidationError if bad

    # single-panel figure keeps the bare figstem; multi-panel gets letters
    labels = [""] if len(contracts) == 1 else _panel_labels(raw_labels)
    return ReasonResult("ok", contracts, labels, reason or "extracted", certainty, raw)


# --- Paper-level scalar extraction (text-only; run once per paper) ----------
# Byte-stable, cached. Resolving q_1c / loading once per paper (rather than
# re-extracting them on every figure) removes the per-figure "extraction
# lottery" where the same paper-level constant is caught on one figure and
# missed on another.
PAPER_SCALAR_INSTRUCTIONS = """You read the methods text and figure captions of \
ONE battery paper and extract two paper-level scalars used to convert its \
voltage curves to areal power vs. time.

Return ONE JSON object, no prose, no markdown fences:
{
  "nominal_capacity_mah_g": <number or null>,  // the specific capacity (mAh/g)
                                        // that a C-rate is defined against for
                                        // this cathode (i.e. 1C); this is the
                                        // theoretical/nominal capacity, however
                                        // the paper phrases it
  "loading_mg_cm2": <number or null>,  // cathode areal mass loading (mg/cm^2)
  "reason": "<short note on where you found them, or why null>"
}

Report a value ONLY if the text states it, explicitly or unambiguously (e.g. a \
loading of "6.4 mg cm-2", or "1C = 200 mAh/g", or "theoretical capacity 274 \
mAh/g"). If a value is not stated, return null for it -- never guess. If the \
paper reports different loadings for different cells, return the one used for \
the standard first-cycle voltage profiles and note the ambiguity in "reason"."""


def _finish_scalars(envelope: dict) -> PaperScalars:
    """Validate a parsed paper-scalar envelope. Raises ValueError on a
    non-numeric or non-positive value so the caller's retry loop can catch it."""
    def positive_or_none(v):
        if v is None:
            return None
        v = float(v)  # raises ValueError on a non-numeric string
        if v <= 0:
            raise ValueError(f"scalar must be positive, got {v}")
        return v

    return PaperScalars(
        nominal_capacity_mah_g=positive_or_none(envelope.get("nominal_capacity_mah_g")),
        loading_mg_cm2=positive_or_none(envelope.get("loading_mg_cm2")),
        reason=str(envelope.get("reason", "") or ""),
    )


def reason_paper_scalars(
    client,
    paper_text: str,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> PaperScalars:
    """Resolve one paper's conversion scalars from its text (no image).

    `paper_text` is the paper's harvested scalar sentences + captions. Makes at
    most two API calls (initial + one corrective retry). On refusal or repeated
    invalid output, returns null scalars with a reason rather than raising -- a
    figure simply stays uncompilable, never guessed.
    """
    system = [{"type": "text", "text": PAPER_SCALAR_INSTRUCTIONS,
               "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": paper_text or "(no text provided)"}]

    last_error = ""
    for _ in range(2):
        response = _create(
            client, model, fallback_model,
            max_tokens=max_tokens, system=system, messages=messages,
        )
        if _refusal_reason(response) is not None:
            return PaperScalars(None, None, "model declined the request")
        raw = _extract_text(response)
        try:
            return _finish_scalars(_parse_envelope(raw))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"That could not be used: {last_error}. Reply again with "
                    "ONLY the JSON object described in the instructions."
                )},
            ]

    return PaperScalars(None, None, f"invalid after retry: {last_error}")
