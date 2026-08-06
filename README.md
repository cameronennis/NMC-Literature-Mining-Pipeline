# voltage-tracer

Digitize **first-cycle** voltage-vs-capacity curves from solid-state NMC
battery-paper figures and convert them to **areal power vs. time**, for
cross-paper comparison across a ~1000-paper corpus.

## The one idea: a hard token boundary

The pipeline is deliberately split so that **exactly one operation costs
tokens** and everything else is deterministic Python you can re-run for free.

```
  intake      PDFs            -> figures/    <stem>.png + <stem>.txt      [free]
  reason      figures         -> contracts/  <stem>.json                 [TOKENS]
  batch       figures+contracts -> traces/   <stem>__<series>.csv        [free]
  compile     traces+contracts  -> compiled/ <stem>__<series>.csv        [free]
```

Only **reason** calls the Anthropic API. It reads a figure image + its caption
and emits a JSON *contract* (plot box, axis tick pixel↔value calibration, and
per-curve colour/label/direction/C-rate). That contract is the sole judgement
the deterministic code can't make itself; given it, the trace/smooth/
calibrate/validate/compile stages need no model.

Within the reason stage, the conversion scalars `q_1c_mah_g` and
`loading_mg_cm2` are **resolved once per paper** (one cheap text-only call that
reads the paper's methods/captions), then used to fill any figure contract that
left them null. They're properties of the electrode, not of a single figure, so
resolving them per-paper stops one figure from missing a value another figure
of the same paper caught. A figure that states its own value still wins.

Because contracts are just JSON on disk, you can re-run tracing and compilation
as many times as you like — tune a threshold, fix a bug — **without paying for
the vision calls again** (`--no-reason`).

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.11. The `anthropic` SDK is only imported by the reason
stage (lazily), so intake/trace/compile work without a key.

## Quickstart

```bash
# Cheap trial: 5 figures, watch the verdicts and cost before committing
voltage-tracer --pdf-dir "Top 21/pdf/Papers" --out-dir run --limit 5

# Full run
voltage-tracer --pdf-dir "Top 21/pdf/Papers" --out-dir run

# Re-trace / re-compile without re-paying for reasoning
voltage-tracer --pdf-dir "Top 21/pdf/Papers" --out-dir run --no-reason

# Reconcile a reused out-dir to the current PDFs (prune stale, skip identical re-renders)
voltage-tracer --pdf-dir "Top 21/pdf/Papers" --out-dir run --fresh
```

**Reusing an `--out-dir`.** By default a run overwrites same-named files but
leaves files from a prior run in place, so pointing a smaller corpus at a
populated folder leaves stale figures/curves behind (and re-reasons them, at
token cost). `--fresh` fixes that: it prunes every artifact whose figure isn't
in the current run and skips re-rendering figures that already exist, so
nothing identical is needlessly rewritten. Manifests are always preserved. One
`--out-dir` per corpus is still the simplest rule; `--fresh` makes reuse safe.

Credentials resolve the standard way (`ANTHROPIC_API_KEY`, or an
`ant auth login` profile) — the reason stage constructs a plain
`anthropic.Anthropic()`.

## Model selection

Set with `--model` (no source edit needed); the default lives at one place,
`reason.py:DEFAULT_MODEL`.

| Model | Notes |
|---|---|
| `claude-fable-5` (default) | Strongest vision; best at reading tick pixels, curve colours, and the legend↔curve mapping — the accuracy bottleneck for the whole project. |
| `claude-opus-4-8` | ~half the cost, close on most figures. |
| `claude-sonnet-5` | Cheapest; fine for clean single-panel figures. |

On a policy refusal, Fable re-runs the request on `--fallback-model` (default
`claude-opus-4-8`) inside the same API call; pass `--fallback-model none` to
disable. The fallback only attaches for Fable/Mythos, the only models that can
return a `refusal` stop reason, so it's a no-op if you switch to Opus/Sonnet.

> ⚠️ **Fable 5 requires 30-day data retention.** Under zero-data-retention it
> returns `400` on every request — check the org setting before a large run.

## Verdicts and scope

The reason stage returns one of four verdicts per figure, all recorded in
`contracts/reason_manifest.json`:

- **ok** — a validated contract is written; the figure flows downstream.
- **skipped** — out of scope. Only **first-cycle** charge/discharge profiles
  are traced; multi-cycle / cycle-fade / rate-over-cycles plots are declined.
- **flagged** — in scope but legend↔curve certainty below `--certainty`
  (default 0.95). No contract written — needs human review.
- **error** — model output couldn't be parsed/validated even after one
  corrective retry. Dropped, never guessed.

A figure with no `q_1c_mah_g` / `loading_mg_cm2` scalars is traced but not
compiled (compile skips it with a recorded reason) — a voltage profile is still
produced, just not the areal-power conversion. The per-paper scalar pass
(above) makes this rare: a figure only stays uncompiled if the paper never
states the value anywhere. Each paper's resolved scalars are recorded under
`paper_scalars` in `reason_manifest.json`, and each figure's verdict lists any
scalar `filled_from_paper`.

## Outputs

```
run/
  figures/    <stem>.png, <stem>.txt, intake_manifest.json
  contracts/  <stem>.json (ok figures only), reason_manifest.json
  traces/     <stem>__<series>.csv (capacity, voltage lineage), batch_summary.json
  compiled/   <stem>__<series>.csv (time_h, power_*_mW_cm2), compile_manifest.json
  run_manifest.json     # ties the four stage manifests together
```

Every stage writes its own manifest and never lets one bad paper halt the run,
so a 1000-paper batch is triaged in one pass.

## Tests

```bash
pytest
```

The suite is fully deterministic and **spends no tokens** — the Anthropic
client is mocked. One live smoke test is gated behind
`ANTHROPIC_API_KEY` + `VT_LIVE=1` and is skipped otherwise.
