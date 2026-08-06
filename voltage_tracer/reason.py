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
DEFAULT_MAX_TOKENS = 2000
DEFAULT_CERTAINTY_THRESHOLD = 0.95

# --- Fixed instruction block (BYTE-STABLE so the prompt cache hits). --------
# Do not interpolate anything volatile (dates, ids, the caption) into this.
INSTRUCTIONS = """You read one figure from a solid-state NMC battery paper and \
emit a JSON contract describing its voltage-vs-capacity curves, for a \
deterministic graph-tracing pipeline.

SCOPE -- FIRST CYCLE ONLY. Only describe the cell's first charge and/or first \
discharge voltage profiles. If the figure is a multi-cycle / cycle-fade / \
rate-capability-over-cycles plot (many overlaid cycles, "cycle number" on an \
axis, capacity-retention curves, etc.) rather than a clean first-cycle voltage \
profile, DECLINE it -- do not invent a contract.

Return ONE JSON object, no prose, no markdown fences:
{
  "decision": "extract" | "skip" | "uncertain",
  "reason": "<one short sentence>",
  "certainty": <number 0..1>,          // your confidence in the legend<->curve
                                        // (colour/label/direction/C-rate) mapping
  "contract": { ... } | null           // required iff decision == "extract"
}

Use "skip" for out-of-scope figures (not a first-cycle voltage profile). Use \
"uncertain" (or certainty below 0.95) when you cannot confidently map each \
curve to its colour, direction, and C-rate. Only use "extract" when you are \
confident.

The "contract" object fields:
  plot_bbox_px: [x0,y0,x1,y1]  plot axes rectangle in image pixels
                               (top-left origin; x1>x0, y1>y0).
  x_axis / y_axis: {
      "scale": "linear" | "log",
      "ticks_px":  [pixel positions of >=2 labelled ticks],
      "ticks_val": [their numeric values, same length]
  }   x is capacity (mAh/g), y is voltage (V). y ticks_px go top-to-bottom so
      pair each pixel with its voltage. Need >=2 DISTINCT ticks per axis.
  series: [ one entry per first-cycle curve:
     {
       "id": "<stable slug, e.g. 1st_charge / 0.2C_discharge>",
       "color_rgb": [r,g,b],          // the curve's colour, 0..255
       "label": "<legend text>",
       "direction": "increasing" | "decreasing",  // charge=increasing V,
                                                   // discharge=decreasing V
       "c_rate": <number or null>,    // e.g. 0.2 for 0.2C; null if unlabelled
       "c_rate_rank": <int or null>   // optional ordering hint, slowest=low
     } ]
  legend_bbox_px: [x0,y0,x1,y1] or null   // legend rectangle to mask, if any
  data_region_frac: [fx0,fy0,fx1,fy1]     // sub-rectangle of the plot box that
                                          // curve data occupies, as fractions
                                          // (default [0,0,1,1]); shrink it to
                                          // exclude in-plot titles/panel letters
                                          // or a legend sitting over the axes.
  q_1c_mah_g: <number or null>       // 1C / theoretical specific capacity, from
                                     // the paper text; null if not stated here.
  loading_mg_cm2: <number or null>   // electrode areal loading, from the text.

Read q_1c_mah_g and loading_mg_cm2 from the supplied caption/methods text when \
present; leave them null if the figure/caption does not state them (the figure \
is then compiled only if the values appear elsewhere). All pixel coordinates \
are in the ORIGINAL image's pixel space."""


@dataclass(frozen=True)
class ReasonResult:
    """Outcome of reasoning over one figure.

    status:
      "ok"      -> `contract` is a validated Contract, ready for the pipeline.
      "skipped" -> out of scope (not a first-cycle voltage profile).
      "flagged" -> in scope but low certainty; needs human review.
      "error"   -> the model's output could not be parsed/validated after a
                   retry, so the figure is dropped (never written as a guess).
    """

    status: str
    contract: Contract | None
    reason: str
    certainty: float | None = None
    raw: str | None = None  # raw model text, kept for debugging failures


@dataclass(frozen=True)
class PaperScalars:
    """Paper-level conversion scalars, resolved once per paper.

    q_1c_mah_g and loading_mg_cm2 are properties of the cathode/electrode, not
    of any single figure, so they are extracted once from the paper's text and
    used to fill any figure contract that left them null (a figure that states
    its own value still wins). Either may be None if the paper doesn't state it.
    """

    q_1c_mah_g: float | None
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
            return ReasonResult("flagged", None, refusal, None, None)
        raw = _extract_text(response)
        last_raw = raw
        try:
            envelope = _parse_envelope(raw)
            return _finish(envelope, certainty_threshold, raw)
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
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
        contract=None,
        reason=f"model output invalid after retry: {last_error}",
        raw=last_raw,
    )


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
        return ReasonResult("skipped", None, reason or "out of scope", certainty, raw)

    if decision == "uncertain":
        return ReasonResult("flagged", None, reason or "model uncertain", certainty, raw)

    if decision != "extract":
        raise ValueError(f"unknown decision {decision!r} (expected extract/skip/uncertain)")

    if certainty is not None and certainty < certainty_threshold:
        return ReasonResult(
            "flagged",
            None,
            reason or f"certainty {certainty} below {certainty_threshold}",
            certainty,
            raw,
        )

    contract_data = envelope.get("contract")
    if not isinstance(contract_data, dict):
        raise ValueError("decision was 'extract' but 'contract' is missing/not an object")

    contract = validate_contract(contract_data)  # raises ValidationError if bad
    return ReasonResult("ok", contract, reason or "extracted", certainty, raw)


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
  "q_1c_mah_g": <number or null>,      // the specific capacity (mAh/g) that
                                        // defines 1C for this cathode
  "loading_mg_cm2": <number or null>,  // cathode areal mass loading (mg/cm^2)
  "reason": "<short note on where you found them, or why null>"
}

Report a value ONLY if the text states it, explicitly or unambiguously (e.g. a \
loading of "6.4 mg cm-2", or "1C = 200 mAh/g"). If a value is not stated, \
return null for it -- never guess. If the paper reports different loadings for \
different cells, return the one used for the standard first-cycle voltage \
profiles and note the ambiguity in "reason"."""


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
        q_1c_mah_g=positive_or_none(envelope.get("q_1c_mah_g")),
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
