"""reason.py tests -- all with a MOCKED client, no live API calls, no tokens."""
import json
from dataclasses import dataclass

from voltage_tracer.contract import Contract
from voltage_tracer.reason import (
    INSTRUCTIONS,
    PAPER_SCALAR_INSTRUCTIONS,
    PaperScalars,
    reason_figure,
    reason_paper_scalars,
)


def _scalars(q=200.0, loading=36.94, reason="from methods"):
    return json.dumps({"q_1c_mah_g": q, "loading_mg_cm2": loading, "reason": reason})


# --- minimal stand-ins for the Anthropic SDK response shape ---


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _StopDetails:
    category: str | None = None


@dataclass
class _Response:
    content: list
    stop_reason: str | None = None
    stop_details: _StopDetails | None = None


class _Refusal:
    """Marker body: makes the fake client return a refusal response."""

    def __init__(self, category="cyber"):
        self.category = category


class FakeClient:
    """Returns canned bodies in sequence and records every create() call.

    Exposes BOTH `messages.create` and `beta.messages.create` so tests can
    assert which endpoint reason.py chose (the beta one carries the
    server-side refusal fallback). `endpoints` records that choice per call.
    """

    def __init__(self, bodies):
        self._bodies = list(bodies)
        self.calls = []
        self.endpoints = []

        outer = self

        class _Messages:
            def __init__(self, endpoint):
                self._endpoint = endpoint

            def create(self, **kwargs):
                outer.calls.append(kwargs)
                outer.endpoints.append(self._endpoint)
                body = outer._bodies.pop(0)
                if isinstance(body, _Refusal):
                    return _Response(
                        content=[], stop_reason="refusal",
                        stop_details=_StopDetails(category=body.category),
                    )
                return _Response(content=[_TextBlock(text=body)], stop_reason="end_turn")

        class _Beta:
            def __init__(self):
                self.messages = _Messages("beta")

        self.messages = _Messages("plain")
        self.beta = _Beta()


# --- a valid contract envelope the model might return ---

_CONTRACT = {
    "plot_bbox_px": [10, 20, 500, 400],
    "x_axis": {"scale": "linear", "ticks_px": [10, 500], "ticks_val": [0, 200]},
    "y_axis": {"scale": "linear", "ticks_px": [400, 20], "ticks_val": [2.5, 4.5]},
    "series": [
        {
            "id": "1st_charge",
            "color_rgb": [0, 0, 0],
            "label": "charge",
            "direction": "increasing",
            "c_rate": 0.2,
        }
    ],
    "q_1c_mah_g": 200.0,
    "loading_mg_cm2": 36.94,
}


def _extract(certainty=0.99, contract=None):
    return json.dumps(
        {
            "decision": "extract",
            "reason": "clean first-cycle profile",
            "certainty": certainty,
            "contract": contract if contract is not None else _CONTRACT,
        }
    )


IMG = "aGVsbG8="  # dummy base64


def test_valid_response_yields_ok_contract():
    client = FakeClient([_extract()])
    res = reason_figure(client, IMG, "image/png", "caption text")
    assert res.status == "ok"
    assert isinstance(res.contract, Contract)
    assert res.contract.series[0].c_rate == 0.2
    assert res.certainty == 0.99
    assert len(client.calls) == 1  # no retry needed


def test_call_carries_image_caption_and_cached_system():
    client = FakeClient([_extract()])
    reason_figure(client, IMG, "image/png", "loading was 36.94 mg/cm2")
    kwargs = client.calls[0]
    # cached, byte-stable instruction block
    assert kwargs["system"][0]["text"] == INSTRUCTIONS
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # user turn carries the image and the caption
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == IMG
    assert content[0]["source"]["media_type"] == "image/png"
    assert "36.94" in content[1]["text"]


def test_skip_decision_is_not_extracted():
    body = json.dumps({"decision": "skip", "reason": "multi-cycle fade plot", "certainty": 0.9})
    client = FakeClient([body])
    res = reason_figure(client, IMG, "image/png", "")
    assert res.status == "skipped"
    assert res.contract is None
    assert "multi-cycle" in res.reason
    assert len(client.calls) == 1


def test_low_certainty_extract_is_flagged():
    client = FakeClient([_extract(certainty=0.80)])
    res = reason_figure(client, IMG, "image/png", "")
    assert res.status == "flagged"
    assert res.contract is None
    assert res.certainty == 0.80


def test_explicit_uncertain_decision_is_flagged():
    body = json.dumps({"decision": "uncertain", "reason": "colours ambiguous", "certainty": 0.6})
    res = reason_figure(FakeClient([body]), IMG, "image/png", "")
    assert res.status == "flagged"
    assert res.contract is None


def test_malformed_then_valid_triggers_one_retry():
    client = FakeClient(["not json at all", _extract()])
    res = reason_figure(client, IMG, "image/png", "")
    assert res.status == "ok"
    assert isinstance(res.contract, Contract)
    assert len(client.calls) == 2
    # the retry fed the bad output + a corrective note back to the model
    retry_msgs = client.calls[1]["messages"]
    assert retry_msgs[-2]["role"] == "assistant"
    assert retry_msgs[-1]["role"] == "user"
    assert "could not be used" in retry_msgs[-1]["content"]


def test_malformed_twice_yields_error_and_no_contract():
    client = FakeClient(["garbage one", "garbage two"])
    res = reason_figure(client, IMG, "image/png", "")
    assert res.status == "error"
    assert res.contract is None
    assert len(client.calls) == 2
    assert res.raw == "garbage two"


def test_invalid_contract_body_retries_then_errors():
    # decision=extract but the contract violates the schema (x1 < x0)
    bad = dict(_CONTRACT)
    bad_env = json.dumps(
        {"decision": "extract", "reason": "x", "certainty": 0.99,
         "contract": {**_CONTRACT, "plot_bbox_px": [500, 20, 10, 400]}}
    )
    client = FakeClient([bad_env, bad_env])
    res = reason_figure(client, IMG, "image/png", "")
    assert res.status == "error"
    assert res.contract is None
    assert len(client.calls) == 2


def test_extract_response_wrapped_in_markdown_fence_is_parsed():
    fenced = "```json\n" + _extract() + "\n```"
    res = reason_figure(FakeClient([fenced]), IMG, "image/png", "")
    assert res.status == "ok"
    assert isinstance(res.contract, Contract)


def test_model_and_threshold_are_honoured():
    client = FakeClient([_extract(certainty=0.90)])
    res = reason_figure(client, IMG, "image/png", "", model="claude-opus-4-8",
                        certainty_threshold=0.85)
    assert res.status == "ok"  # 0.90 clears the lowered 0.85 bar
    assert client.calls[0]["model"] == "claude-opus-4-8"


# --- model selection / Fable-specific handling ---


def test_fable_default_uses_beta_endpoint_with_opus_fallback():
    # DEFAULT_MODEL is Fable 5: beta endpoint carrying the Opus-4.8 fallback.
    client = FakeClient([_extract()])
    reason_figure(client, IMG, "image/png", "")
    assert client.endpoints == ["beta"]
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-fable-5"
    assert kwargs["betas"] == ["server-side-fallback-2026-06-01"]
    assert kwargs["fallbacks"] == [{"model": "claude-opus-4-8"}]


def test_non_fable_model_uses_plain_endpoint_without_fallback():
    for model in ("claude-opus-4-8", "claude-sonnet-5"):
        client = FakeClient([_extract()])
        reason_figure(client, IMG, "image/png", "", model=model)
        assert client.endpoints == ["plain"]
        assert "fallbacks" not in client.calls[0]
        assert "betas" not in client.calls[0]


def test_fallback_can_be_disabled_for_fable():
    client = FakeClient([_extract()])
    reason_figure(client, IMG, "image/png", "", model="claude-fable-5", fallback_model=None)
    assert client.endpoints == ["plain"]
    assert "fallbacks" not in client.calls[0]


def test_no_thinking_parameter_is_sent():
    # Fable rejects any explicit thinking config with a 400.
    client = FakeClient([_extract()])
    reason_figure(client, IMG, "image/png", "")
    assert "thinking" not in client.calls[0]


def test_refusal_is_flagged_not_crashed():
    # HTTP 200 + stop_reason "refusal" + EMPTY content: reading content[0]
    # would raise, so the refusal must be detected first.
    client = FakeClient([_Refusal(category="cyber")])
    res = reason_figure(client, IMG, "image/png", "")
    assert res.status == "flagged"
    assert res.contract is None
    assert "declined" in res.reason and "cyber" in res.reason
    assert len(client.calls) == 1  # a refusal is not retried


# --- paper-level scalar extraction ---


def test_paper_scalars_parsed_from_json():
    client = FakeClient([_scalars(q=180.0, loading=6.4)])
    ps = reason_paper_scalars(client, "methods text", model="claude-opus-4-8")
    assert isinstance(ps, PaperScalars)
    assert ps.q_1c_mah_g == 180.0 and ps.loading_mg_cm2 == 6.4


def test_paper_scalars_call_is_text_only_with_cached_system():
    client = FakeClient([_scalars()])
    reason_paper_scalars(client, "the paper text", model="claude-opus-4-8")
    kwargs = client.calls[0]
    assert kwargs["system"][0]["text"] == PAPER_SCALAR_INSTRUCTIONS
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # text-only: user content is a plain string, no image block
    assert kwargs["messages"][0]["content"] == "the paper text"


def test_paper_scalars_null_when_not_stated():
    body = json.dumps({"q_1c_mah_g": None, "loading_mg_cm2": None, "reason": "not stated"})
    ps = reason_paper_scalars(FakeClient([body]), "text", model="claude-opus-4-8")
    assert ps.q_1c_mah_g is None and ps.loading_mg_cm2 is None
    assert "not stated" in ps.reason


def test_paper_scalars_reject_nonpositive_then_retry():
    bad = json.dumps({"q_1c_mah_g": -5, "loading_mg_cm2": 6.4})
    client = FakeClient([bad, _scalars(q=180.0, loading=6.4)])
    ps = reason_paper_scalars(client, "text", model="claude-opus-4-8")
    assert ps.q_1c_mah_g == 180.0
    assert len(client.calls) == 2


def test_paper_scalars_malformed_twice_returns_nulls():
    client = FakeClient(["garbage", "still garbage"])
    ps = reason_paper_scalars(client, "text", model="claude-opus-4-8")
    assert ps.q_1c_mah_g is None and ps.loading_mg_cm2 is None
    assert "invalid after retry" in ps.reason


def test_paper_scalars_refusal_returns_nulls():
    ps = reason_paper_scalars(FakeClient([_Refusal()]), "text", model="claude-opus-4-8")
    assert ps.q_1c_mah_g is None and ps.loading_mg_cm2 is None
    assert "declined" in ps.reason


def test_paper_scalars_fable_default_uses_beta_endpoint():
    client = FakeClient([_scalars()])
    reason_paper_scalars(client, "text")  # DEFAULT_MODEL = fable
    assert client.endpoints == ["beta"]
    assert client.calls[0]["fallbacks"] == [{"model": "claude-opus-4-8"}]
