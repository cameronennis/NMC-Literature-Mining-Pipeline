# Voltage Tracer — checklist / backlog

Priority now: **reasoning-stage + tracing accuracy/consistency**. Features below are
deferred until that's solid.

## Now — accuracy / consistency
- [ ] **Terminal-drop across large mask gaps.** Near-vertical drops thinned to nothing over
      several columns aren't bridged (`ext_run_gap_px` / `ext_col_offset` too small).
- [ ] **Same-colour extended overlap.** Bridging assumes one charge + one discharge per
      colour; two same-colour curves overlapping long aren't separated.
- [ ] **P16 → 0 figures.** Intake caption detector misses the split "FIGURE 1" style label.
- [ ] **Over-large `legend_bbox_px`.** Symptom fixed (legend mask spares a transiting curve);
      optionally tighten the reason instructions to draw the box snugly at the source.
- [ ] **Dashed / non-solid lines.** Re-check if a non-solid filter is needed. (Colour-snap
      optional/off; beam/bidirectional not worth rebuilding.)
- [x] **P13 fig8a (-20C) tracking failure.** Was 2-15px stubs; fixed by hue-recover color
      masking (2026-08-14) — the -20C strokes are desaturated/dark and the Lab-distance mask
      dropped them. All four curves now trace full-range.
- [x] **Coloured annotations traced as curves (fig1a/1c "79.0%" labels+arrows).** The red/blue
      %-labels and their horizontal arrows share a series colour, so achromatic glyph detection
      missed them and the discharge jogged sideways onto them at the steep drop. Fixed
      (2026-08-14) by `colored_annotation_mask`: per-colour detection of glyph clusters AND long
      thin horizontal rules, masking only SMALL isolated components so a curve / its drop (one
      large component) is spared. Discharge now drops cleanly.
- [x] **Steep-drop kink onto annotation (fig1a).** Even with the label masked, the raw trace
      still jogged off the near-vertical drop (a huge second derivative: slope +9.9 -> -4.4).
      Fixed (2026-08-14) by `_trim_kink`: detect a near-vertical descent that abruptly REVERSES
      and truncate to the last steep point, so `extend_terminal_drops` follows the true slope
      down. Both fig1a discharges now reach V2.42 (the real drop foot). A real drop bottom only
      decelerates (no reversal) and a gentle curve never hits the steep threshold, so both are
      untouched; A/B shows no other figure affected.
- [x] **Different-colour occlusion (near-coincident cells).** fig1a/fig1c: ZrO2 (red) ~
      coincides with bare NCM911 (blue), so each red curve was repeatedly occluded by its blue
      twin and the tracker deactivated it at the crossover (charge died cap 85, discharge 108).
      Fixed by occlusion-aware coasting (2026-08-14): a track coasts on prediction through a
      column covered by another colour, then re-acquires; trailing coasts-into-nothing are
      trimmed before the coherence gate. Red now traces full-range. (Minor: fig8a blue
      discharge is sparser, n164->82, same extent/coherent.)

## Later — gravimetric fallback
- [ ] **mW/g route when mW/cm² impossible.** Specific x-axis + no loading: `t = spec_cap /
      (c_rate·Q_nom,spec)`, `P = V·c_rate·Q_nom,spec` [mW/g]. Prefer areal, fall back to
      specific. Emit mW/g in a **new power column** + a `power_basis` tag. (Loading cancels in
      `t`, so time is unaffected.)
- [ ] **"Various loadings" fallback guard.** No stated nominal capacity + differing per-series
      loadings → don't share one curve's areal capacity across all; refuse or flag.

## Later — per-curve metadata master table
Goal: every traced curve carries filterable attributes; a separate plotting program reads
the table to choose which curves to plot.
- [ ] **Shape.** One row per curve, keyed by trace id `<figstem>__<series>`, grouped by source
      paper. Compiled CSVs stay pure numeric data; join on the id.
- [ ] **Columns.** Identity/grouping: paper, figure, panel, series, direction. Filter
      attributes: c_rate, power_basis, cathode composition, electrolyte, temperature,
      pressure, mass loading. Each inferred value gets a `<field>_assumed` flag beside it.
      (Nominal capacity is NOT here — it's a compile-internal input, consumed before the
      plotting values exist.) Power is always normalized per area (mW/cm²) or per mass (mW/g).
- [ ] **Population / cost control.**
      - Free from contract: paper/figure/series/direction/c_rate/power_basis; **composition
        from the series label**.
      - Fold temperature / electrolyte / pressure / loading / nominal capacity into the
        **existing per-paper text pass** (`reason_paper_scalars`) — one enlarged call per
        paper, reused across its curves, **no new API calls**.
      - Assume defaults when unstated (e.g. temperature = RT), only when the paper doesn't
        emphasise that variable; mark via the `_assumed` flag.
- [ ] **Output.** `curves_metadata.csv` per run; consumed downstream (pipeline doesn't
      filter). Should slot into the existing `Data Findings/SS_NMC_master_extraction` workflow.

## Housekeeping
- [ ] **`--fresh` prunes panel derivatives** (keys on intake figstems → deletes `<fig><letter>`
      outputs). Fix before using `--fresh` on multi-panel runs.
- [ ] **Overwritten P13 fig1c files (2026-08-13).** 5 fixed-version CSV/summary files replaced
      the originals in `Updated Output/traces/` (unrecoverable). Decide: keep, or delete so the
      run simply lacks P13 fig1c.
