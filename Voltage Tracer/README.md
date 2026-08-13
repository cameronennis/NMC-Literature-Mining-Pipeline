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
per-curve colour/label/direction plus how each curve's current is set). That
contract is the sole judgement the deterministic code can't make itself; given
it, the trace/smooth/calibrate/validate/compile stages need no model.

### Canonical areal basis

Compile normalises every curve to **one basis** — areal capacity (mAh/cm²) and
areal current density `j` (mA/cm²) — then applies a single formula:
`t = areal_capacity / j`, `P = voltage · j`. Two bases are supported, `specific`
(per gram) and `areal` (per area), bridged by the electrode loading
(`areal = specific · loading / 1000`). Absolute capacity (mAh) is out of scope.

Per curve, `j` comes from either a **directly-stated current**
(`current_value` + `current_basis`) or a **C-rate**. A C-rate needs a nominal
reference capacity to become a current (`j = c_rate · Q_ref`); compile resolves
`Q_ref` **once per panel**: from a stated theoretical/nominal capacity if the
paper gives one, else — flagged low-confidence — from the panel's slowest
(≤ 1C) curve's delivered capacity. A curve whose `j` can't be resolved is
skipped, never guessed.

Loading may vary **per curve** (`series.loading_mg_cm2`), for "various mass
loadings" figures where each curve is a different-loading cell; otherwise the
figure-level `loading_mg_cm2` applies. The two paper-level constants
(`loading_mg_cm2` and the nominal specific capacity) are **resolved once per
paper** (one cheap text-only call over the methods/captions), then used to fill
any figure contract that left them null. They're properties of the electrode,
not of a single figure, so resolving them per-paper stops one figure from
missing a value another figure of the same paper caught. A figure that states
its own value still wins.

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
voltage-tracer --pdf-dir <paper pdf directory> --out-dir run --limit 5

# Full run
voltage-tracer --pdf-dir <paper pdf directory> --out-dir run

# Re-trace / re-compile without re-paying for reasoning
voltage-tracer --pdf-dir <paper pdf directory> --out-dir run --no-reason

# Reconcile a reused out-dir to the current PDFs (prune stale, skip identical re-renders)
voltage-tracer --pdf-dir <paper pdf directory> --out-dir run --fresh
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

- **ok** — a validated contract is written per in-scope panel; the figure flows
  downstream.
- **skipped** — out of scope. In scope: **first-cycle** *and* **rate-capability**
  charge/discharge profiles (one curve per C-rate). Declined: figures that show
  **cycle decay** — the same cell/rate at several cycle numbers (cycle 1 vs 50
  vs 100…), capacity-retention/cycling plots. The test: curves differ by C-rate
  or cell → keep; differ by cycle number → decline.
- **flagged** — in scope but legend↔curve certainty below `--certainty`
  (default 0.95). No contract written — needs human review.
- **error** — model output couldn't be parsed/validated even after one
  corrective retry. Dropped, never guessed.

**Multi-panel figures are split.** A figure with several panels (`a`, `b`, `c`…)
yields one contract per in-scope voltage-profile panel — `<figstem>a.json`,
`<figstem>b.json`, each paired with a `<figstem>a.png` hardlink of the figure —
so each panel traces independently. Out-of-scope panels (CV, rate bars, Nyquist,
SEM) are omitted. A single-panel figure keeps the bare `<figstem>`.

A curve whose areal current can't be resolved (no loading to bridge a specific
axis, or a C-rate with no reference capacity and no ≤1C sibling) is traced but
not compiled — compile skips it with a recorded reason, so a voltage profile is
still produced, just not the areal-power conversion. The per-paper scalar pass
(above) makes this rare. Each paper's resolved scalars are recorded under
`paper_scalars` in `reason_manifest.json`, and each figure's verdict lists any
scalar `filled_from_paper`. In `compile_manifest.json`, each compiled curve
records its `reference_source` (`stated_theoretical` vs `fallback_slowest_curve`)
and the resolved `q_ref_areal_mah_cm2`; fallback-derived curves also carry a
`reference_low_confidence` flag column in their compiled CSV.

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
