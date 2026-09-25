"""Universal, dimensionless tuning constants for the tracer.

Every value here is a multiple of, or fraction of, a quantity *measured
from the figure at runtime* (`metrics.FigureMetrics`) -- never an absolute
pixel count tuned to one figure. That is the whole point: these numbers are
meant to hold across ~1000 heterogeneous figures, because the thing they
scale (`line_width_px`, `plot_w_px`, `intercurve_color_sep`, ...) is
re-measured per figure.

Naming convention for the suffix:
- `_lw`     : multiple of `line_width_px` (a "how many stroke-widths" ruler)
- `_wfrac`  : fraction of `plot_w_px`   (a horizontal span, resolution-free)
- `_hfrac`  : fraction of `plot_h_px`   (a vertical span, resolution-free)
- `_sepfrac`: fraction of `intercurve_color_sep` (a color-distance ruler)
- `_area_k` : multiple of `line_width_px**2` (an area ruler)

If you find yourself wanting to add a raw pixel literal anywhere else in
the codebase, add a named dimensionless constant here instead and multiply
it by the appropriate measured metric.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    # --- preprocess ---
    # speckle floor: reject blobs below ~this many stroke-width boxes.
    # line_width**2 ~ one stroke-width box; JPEG halo is a fraction of a
    # box, a real (even crossing-chopped) curve fragment is several. Tuned
    # to reproduce the validated ~8px floor at line_width 2 (2.0 * 2**2).
    speckle_area_k: float = 2.0
    # color-mask tolerance as a fraction of the nearest color separation,
    # so a mask can never reach halfway to a neighbouring series' color.
    color_tol_sepfrac: float = 0.45
    # absolute cap (encoded-Lab units) when there is no neighbour to bound
    # against (single-color figure) -- generous enough for anti-aliasing.
    color_tol_abs_cap: float = 45.0
    # Terminal-drop lenient mask. Deliberately generous -- a steep thin
    # fall desaturates so hard on JPEG that its OWN pixels land farther from
    # its color than a neighbouring series' color does (measured on the 20C
    # navy drop: median 44 from navy vs 25 from the 1C blue). So this mask
    # CANNOT be color-bounded below the nearest neighbour without failing to
    # descend the drop. It is used only by `extend_terminal_drops`, which
    # chains fragments SPATIALLY from the track's verified endpoint -- that
    # spatial continuity, not color, is what prevents grabbing a neighbour
    # (the 1C blue never sits where the 20C drop is). A multiple of the
    # color separation, capped absolutely.
    lenient_tol_sepfrac: float = 1.6
    lenient_tol_abs_cap: float = 70.0
    # a gridline/axis spans essentially the whole plot; a curve plateau does
    # not. Fraction of the run's own dimension. (Scale-invariant already.)
    gridline_span_frac: float = 0.85

    # --- skeletonize ---
    # two skeleton pixels within a stroke-width belong to one cluster.
    merge_gap_lw: float = 1.0

    # --- tracker: column-count windows (fractions of plot width) ---
    # slope window must span enough of the curve to stay a stable predictor
    # THROUGH a shallow same-color crossing: too short and, on a narrow
    # figure, the fit collapses to a flat plateau's last few points and the
    # track grabs the wrong branch after the merge. ~3.5% of width (=~8-10
    # mAh/g here) is robust on both the wide 8-rate figure and the narrow
    # 2-color panel.
    slope_window_wfrac: float = 0.035   # trailing slope-fit span
    init_wfrac: float = 0.016           # columns used to seed tracks
    link_gap_wfrac: float = 0.006       # max column gap when chaining seeds
    confirm_run_wfrac: float = 0.010    # run length before a pending track confirms
    # coast-to-deactivate must outlast the longest same-color merge (where a
    # starved track gets no candidate). A merge's column-length is a local
    # geometric feature, only weakly tied to plot width, so this fraction is
    # generous.
    deactivate_wfrac: float = 0.080     # missed-column coast before a track dies
    trim_window_wfrac: float = 0.030    # sparse-tail density window
    head_probe_wfrac: float = 0.012     # points used to test a leading outlier

    # --- tracker: vertical tolerances ---
    # These are vertical DISTANCES in the plot, so they scale with plot
    # HEIGHT (a curvature/drift tolerance grows with the figure), not with
    # stroke width. Fractions of plot_h_px.
    max_jump_hfrac: float = 0.037      # per-column prediction-match gate
    reversal_tol_hfrac: float = 0.030  # cumulative against-direction drift allowed
    head_residual_hfrac: float = 0.030 # leading point counts as outlier past this
    # chaining tolerance is a SLOPE (y-change per column). A px/column slope
    # is already resolution-invariant, so express it per stroke width.
    link_tol_lw: float = 4.0

    # --- tracker: occlusion-aware coasting ---
    # A curve that another colour is drawn over for a stretch (two near-
    # coincident cells, e.g. coated vs bare) vanishes from its own mask and,
    # untreated, the online tracker deactivates it as if it ended. When the
    # column it WOULD occupy is covered by another colour, coast the track on
    # its prediction instead of counting a miss, so it stays alive and re-
    # acquires when it reappears -- bounded to this fraction of plot width so a
    # curve that genuinely ends (no occlusion) still deactivates promptly. The
    # coasted points are marked unobserved (down-weighted, filterable).
    coast_max_wfrac: float = 0.16

    # --- tracker: terminal-drop extension ---
    ext_run_gap_hfrac: float = 0.15    # vertical occlusion tolerated between fragments
    ext_min_extension_hfrac: float = 0.030  # minimum total descent to commit
    ext_col_offset_lw: float = 2.0     # how many columns ahead a drop fragment may sit
    ext_step_lw: float = 1.5           # densify spacing along the drop
    ext_min_advance_lw: float = 1.0    # a fragment must advance the frontier by this
    ext_back_overlap_lw: float = 3.0   # allowed backward overlap when picking a fragment

    # --- tracker: bridging occlusion-split segments ---
    # After tracking, a curve hidden under another colour for a long stretch
    # dies and its visible continuation is left orphaned. `bridge_occluded_
    # segments` stitches a downstream same-colour, same-DIRECTION chain onto the
    # track across the gap, but ONLY when the gap is filled by another curve's
    # colour (proof it was an occlusion, not a real end) and the join is gentle.
    bridge_min_gap_wfrac: float = 0.015   # a bridge must span at least this gap
                                          # (shorter splits are the tracker's job)
    bridge_overlap_frac: float = 0.5      # >= this fraction of the gap columns must
                                          # carry another colour near the bridge path
    bridge_max_slope: float = 3.0         # max |dy/dx| of the straight bridge, a pure
                                          # pixel ratio (a plateau/knee is < ~2; a
                                          # terminal drop is 5-20 -> the wrong segment)
    bridge_min_chain_wfrac: float = 0.020 # a continuation chain must span this wide

    # --- tracker: coherence gate (reject a scattered, non-curve track) ---
    # A finished track whose raw trace departs from the best MONOTONE curve in
    # its declared direction by more than this fraction of its own vertical
    # span is not a real voltage curve -- it is a flat, scattered overlap smear
    # the tracker latched onto (e.g. a heavily-occluded high-rate charge that
    # has no clean rising line). It is dropped and its series offered back to
    # the pending pool so a cleaner branch can be tried in its place. The metric
    # is a direction-aware isotonic residual / span, so a genuine S-shaped
    # (curved but monotone) profile scores ~0 and is never rejected; only a
    # trace that wanders AGAINST its declared direction scores high (a chaotic
    # band scores ~0.3, clean curves <0.01).
    coherence_max_scatter: float = 0.12
    # Only judge coherence for a track that spans at least this fraction of the
    # plot width -- enough to be a real curve. Shorter fragments/stubs (a failed
    # trace in a low-capacity tangle) are left to the other trims: their tiny
    # span makes the ratio metric unreliable, and rejecting them is a separate
    # concern from "a full curve was traced from chaos".
    coherence_min_span_wfrac: float = 0.10
    # When a reclaimed curve starts mid-plot (its early part was consumed by a
    # rejected incoherent track), scan back over the columns to its left and
    # re-attach the branch that continues its trend -- "rescan that x range for
    # more valid curves". The lead-in follows an ALREADY-identified branch, and
    # a monotone-side filter keeps it off curves lying the other way, so it can
    # tolerate a looser per-column jump than forward tracking: this multiple of
    # the normal jump gate. It stops once no on-direction candidate is found.
    lead_extend_jump_mult: float = 2.0

    # --- kink trim (a steep run that abruptly reverses -- jog onto an annot) ---
    # In a near-vertical terminal drop the raw slope is steep and consistent; a
    # jog onto a horizontal annotation (or wrong branch) reverses it abruptly --
    # a very large second derivative. Detect a steep descent (|dy/dx| in the
    # declared direction >= kink_steep_slope, i.e. near-vertical) immediately
    # followed by a REVERSAL (slope <= kink_collapse_slope, moving back against
    # the direction) and truncate to the last steep point, so the terminal-drop
    # pass then follows the true slope down. A real curve turns gradually (small
    # second derivative) and a real drop bottom just decelerates (never
    # reverses), so both are untouched.
    kink_steep_slope: float = 2.75    # dy/dx ~ tan(70deg): near-vertical
    kink_collapse_slope: float = -0.5  # a clear reversal against the direction

    # --- annotation-flat trim (a horizontal figure line mistraced as curve) ---
    # A curve that abruptly flattens (a CUSP into a dead-horizontal run) and
    # stays flat for a long stretch has almost certainly latched onto a figure
    # annotation -- an underline, a retention arrow, a legend rule -- not a real
    # plateau (which slopes across such a span and is approached gradually). The
    # flat run is excised; points after it are kept only if they continue the
    # pre-flat trajectory (so a curve resuming past the line survives).
    flat_band_lw: float = 0.5          # "horizontal" = pixel-y stays within this
                                       # many stroke widths across the WHOLE run --
                                       # a DRAWN line is dead flat (~0 drift); even a
                                       # gentle real plateau (or a high-rate curve's
                                       # brief flat start) wobbles more than this over
                                       # a 5%-width span
    flat_min_wfrac: float = 0.05       # ...for at least this fraction of plot width
    flat_cusp_band_lw: float = 3.0     # entered via a cusp: the curve was this many
                                       # stroke widths off the flat level just before

    # --- pending-pool / misc counts ---
    pending_gap_wfrac: float = 0.006   # max column gap when growing a pending run
    pending_drop_cols: int = 3         # confirm within this many missed cols or drop
    min_chain_points: int = 3          # seed chain must have at least this many pts

    # --- text-region auto-detection (fractions of plot dims / Lab units) ---
    text_chroma_max: float = 22.0      # a glyph is near-neutral in Lab chroma
    text_dark_lfrac: float = 0.6       # and darker than this fraction of full L
    # COLOURED annotations (a red "79.0%" arrow-label, a blue "73.5%") share a
    # series' colour, so the achromatic text pass misses them and a curve then
    # traces their strokes. `colored_annotation_mask` handles them PER colour
    # (so differently-coloured dashes never fuse) and ONLY on small isolated
    # components -- a curve, or its steep terminal drop, is one large connected
    # component and is always spared. Two kinds are masked: glyph clusters
    # (labels) and long thin horizontal rules/arrows.
    annot_detect_colored: bool = True
    annot_curve_span_frac: float = 0.5   # a component spanning >= this frac of
                                         # plot width OR height is a curve: spare it
    annot_rule_min_wfrac: float = 0.10   # a rule/arrow spans >= this frac of width
    annot_rule_max_lw: float = 2.5       # ...and is <= this many stroke widths thick
    glyph_min_hfrac: float = 0.012     # glyph height band, fraction of plot height
    glyph_max_hfrac: float = 0.10
    glyph_max_wfrac: float = 0.10      # a single glyph is not very wide
    glyph_min_extent: float = 0.15     # glyphs are reasonably filled
    glyph_gap_wfrac: float = 0.03      # glyphs in a word/line sit within this gap
    min_glyphs_per_cluster: int = 2    # a lone blob is not text
    text_line_height_k: float = 2.0    # a text line spans ~this many glyph heights;
                                       # taller fused blobs are curve fragments
    # A curve segment chopped at a crossing can look glyph-like, but its ink is
    # one THIN DIAGONAL stroke, not a 2D horizontal text cluster. Skip a
    # candidate region whose ink principal axis is diagonal (|slope| >= this)
    # AND whose perpendicular spread is only ~one stroke thick (<= thin_lw *
    # line_width). Real text is near-horizontal (|slope| ~ 0) or clearly thicker.
    text_line_diag_min_slope: float = 0.20   # ~11deg; well above rotated-text drift
    text_line_thin_lw: float = 1.5           # perpendicular spread, in stroke widths
    text_pad_lw: float = 2.0           # pad detected text bbox by a few stroke widths
    swatch_gap_lw: float = 3.0         # max gap (in stroke widths) when walking a
                                       # text region outward across a legend swatch
    swatch_flat_lw: float = 2.5        # a swatch's colored pixels stay within this
                                       # many stroke widths in y (a curve spans more)

    # --- smooth.py: non-destructive trace smoothing (all dimensionless) ---
    smooth_k: float = 1.0              # spline strength: s = n*(k*sigma)^2
    smooth_sigma_floor_lw: float = 0.4  # sigma floored at this * line_width_px
    smooth_endpoint_w: float = 1e3     # endpoint weight multiplier (pins ends)
    # per-point confidence factors (each clamped to [floor, 1]); low weight =
    # smooth freely through this point
    w_cost_lw: float = 4.0             # match cost normalized by this * line_width
    w_bridged_floor: float = 0.2       # weight of a prediction-bridged point
    w_cluster_floor: float = 0.3       # weight when a column is fully merged
    w_cost_floor: float = 0.2          # weight of a very-high-cost point
    w_floor: float = 0.15              # overall floor on combined weight

    # --- validate.py: flag thresholds (all dimensionless) ---
    mono_tol_lw: float = 1.0           # allowed reversal, in stroke widths (sub-stroke = noise)
    mono_min_run: int = 3              # a reversal must span >= this many points to flag
    capacity_order_tol_frac: float = 0.05  # capacity-ordering slack, fraction of range
    smooth_deviation_k: float = 3.0    # flag |raw-smooth| beyond this * measured sigma
    deviation_min_run: int = 3         # over at least this many points


DEFAULTS = Defaults()
