import cv2
import numpy as np
import pytest

from voltage_tracer.preprocess import (
    color_mask,
    crop_to_plot_area,
    load_image,
    mask_gridlines_and_axes,
    mask_legend_region,
    mask_outside_data_region,
    mask_regions,
    remove_small_components,
)


def _blank(h=300, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


# --- load_image ---


def test_load_image_roundtrips_rgb(tmp_path):
    img = _blank()
    img[10:20, 30:40] = [255, 0, 0]  # pure red block, RGB
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    loaded = load_image(path)
    assert loaded.shape == img.shape
    assert tuple(loaded[15, 35]) == (255, 0, 0)


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_load_image_undecodable_file_raises(tmp_path):
    path = tmp_path / "junk.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="could not decode"):
        load_image(path)


# --- crop_to_plot_area ---


def test_crop_returns_exact_region():
    img = _blank()
    img[50, 60] = [1, 2, 3]
    crop = crop_to_plot_area(img, (60, 50, 160, 150))
    assert crop.shape == (100, 100, 3)
    assert tuple(crop[0, 0]) == (1, 2, 3)


def test_crop_clamps_to_image_bounds():
    img = _blank(h=100, w=100)
    crop = crop_to_plot_area(img, (-20, -20, 250, 250))
    assert crop.shape == (100, 100, 3)


def test_crop_outside_image_raises():
    img = _blank(h=100, w=100)
    with pytest.raises(ValueError, match="does not overlap"):
        crop_to_plot_area(img, (500, 500, 600, 600))


# --- mask_gridlines_and_axes ---


def _figure_with_gridlines_and_curve():
    """White canvas with black axes, gray horizontal gridlines, and a red
    anti-aliased curve."""
    img = _blank(h=300, w=400)
    cv2.rectangle(img, (20, 20), (380, 280), (0, 0, 0), 2)  # axes box
    for y in (85, 150, 215):  # gridlines
        cv2.line(img, (22, y), (378, y), (180, 180, 180), 1)
    pts = np.array(
        [[x, int(250 - 180 * (x - 30) / 350 + 20 * np.sin(x / 40))]
         for x in range(30, 370)]
    )
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(p0), tuple(p1), (220, 30, 30), 2, cv2.LINE_AA)
    return img, pts


def test_gridline_mask_covers_gridlines():
    img, _ = _figure_with_gridlines_and_curve()
    mask = mask_gridlines_and_axes(img)
    # sample the middle of each gridline
    for y in (85, 150, 215):
        assert mask[y, 100:300].mean() > 0.8, f"gridline at y={y} not masked"
    # and the axes box edges
    assert mask[20, 100:300].mean() > 0.8
    assert mask[100:250, 20].mean() > 0.8


def test_gridline_mask_spares_most_of_the_curve():
    img, pts = _figure_with_gridlines_and_curve()
    mask = mask_gridlines_and_axes(img)
    on_curve = np.array([mask[y, x] for x, y in pts])
    assert on_curve.mean() < 0.2, "curve pixels wrongly masked as gridlines"


def test_gridline_mask_tolerates_slight_skew():
    img = _blank(h=300, w=400)
    # a "horizontal" gridline with ~1 degree of skew
    cv2.line(img, (20, 150), (380, 156), (100, 100, 100), 1)
    mask = mask_gridlines_and_axes(img)
    ys = np.round(np.linspace(150, 156, 361)).astype(int)
    hits = np.array([mask[y, x] for x, y in zip(range(20, 381), ys)])
    assert hits.mean() > 0.8


def test_gridline_mask_spares_black_curve_plateau():
    """A BLACK curve's flat plateau is achromatic, so the saturation guard
    can't protect it -- the length gate must: a plateau spanning ~60% of
    the plot is below the near-full-span requirement for gridlines."""
    img = _blank(h=300, w=400)
    cv2.rectangle(img, (10, 10), (390, 290), (0, 0, 0), 2)  # axis box
    cv2.line(img, (30, 150), (270, 148), (20, 20, 20), 2)   # 60%-width plateau
    mask = mask_gridlines_and_axes(img)
    assert not mask[147:152, 40:260].any(), "black plateau eaten by gridline mask"
    assert mask[10, 100:300].mean() > 0.8, "axis box no longer masked"


def test_gridline_mask_spares_colored_plateau():
    """A colored curve's long flat plateau is straight enough to trip the
    Hough detector, but must not be masked -- only achromatic (gray/black)
    pixels count as gridlines."""
    img = _blank(h=300, w=400)
    cv2.line(img, (20, 150), (380, 150), (220, 30, 30), 2)  # red plateau
    mask = mask_gridlines_and_axes(img)
    # no pixel that is actually red may be masked (adjacent white background
    # pixels getting masked is harmless -- they belong to no curve)
    is_red = (img[..., 0] > 180) & (img[..., 1] < 100)
    assert not (mask & is_red).any()


# --- mask_legend_region ---


def test_legend_override_bbox_masks_exactly_that_region():
    img = _blank()
    mask = mask_legend_region(img, override_bbox=(300, 30, 380, 90))
    assert mask[30:90, 300:380].all()
    mask_outside = mask.copy()
    mask_outside[30:90, 300:380] = False
    assert not mask_outside.any()


def test_legend_override_spares_transiting_curve():
    # an over-large legend box that a colored curve passes through: the curve
    # (a tall saturated stroke spanning the box) is spared, while a short
    # swatch and the empty box area stay masked.
    img = _blank()
    img[30:90, 348:351] = (30, 30, 200)   # vertical blue curve through the box
    img[40:43, 305:325] = (30, 30, 200)   # a short blue swatch inside the box
    mask = mask_legend_region(img, override_bbox=(300, 30, 380, 90))
    assert not mask[60, 349], "a curve transiting the legend box must be spared"
    assert mask[41, 315], "a short swatch must stay masked"
    assert mask[80, 360], "empty legend area must stay masked"


def test_legend_heuristic_finds_swatch_with_text():
    img = _blank()
    # swatch: short saturated red dash; "text": dark blob to its right
    cv2.line(img, (300, 50), (320, 50), (230, 20, 20), 3)
    cv2.putText(img, "0.5C", (326, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    mask = mask_legend_region(img)
    assert mask[45:56, 300:330].any(), "legend swatch region not masked"


def test_legend_heuristic_spares_long_curves():
    img = _blank()
    pts = np.array(
        [[x, int(150 + 60 * np.sin(x / 50))] for x in range(20, 380)]
    )
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(p0), tuple(p1), (230, 20, 20), 2, cv2.LINE_AA)
    mask = mask_legend_region(img)
    on_curve = np.array([mask[y, x] for x, y in pts])
    assert on_curve.mean() < 0.1, "curve wrongly masked as legend"


# --- mask_outside_data_region ---


def test_data_region_excludes_top_and_right_bands():
    mask = mask_outside_data_region((200, 400), (0.0, 0.1, 0.84, 1.0))
    assert mask[0:20, :].all(), "top 10% not excluded"
    assert mask[:, 336:].all(), "right 16% not excluded"
    assert not mask[100, 100], "interior wrongly excluded"
    assert not mask[199, 0], "bottom-left corner (in-region) excluded"


def test_data_region_full_box_masks_nothing():
    assert not mask_outside_data_region((50, 60), (0.0, 0.0, 1.0, 1.0)).any()


def test_mask_regions_covers_listed_boxes_only():
    mask = mask_regions((100, 200), [(10, 20, 40, 60), (150, 5, 190, 25)])
    assert mask[20:60, 10:40].all() and mask[5:25, 150:190].all()
    assert not mask[0, 0] and not mask[80, 100]
    assert mask_regions((100, 200), []).sum() == 0


# --- remove_small_components ---


def test_remove_small_components_keeps_fragmented_curve_pieces():
    """A curve chopped into ~15px pieces by overlapping curves must
    survive -- the floor only clears 1-4px JPEG speckle."""
    mask = np.zeros((100, 200), dtype=bool)
    for x0 in (10, 40, 70, 100):  # four short fragments, ~15px each
        mask[50:52, x0 : x0 + 12] = True
    cleaned = remove_small_components(mask, min_area=8)
    assert cleaned.sum() == mask.sum(), "genuine curve fragments removed"


# --- old remove_small_components speckle test ---


def test_remove_small_components_drops_speckle_keeps_curve():
    mask = np.zeros((200, 300), dtype=bool)
    mask[100:103, 10:290] = True  # long curve stroke (~840 px)
    rng = np.random.default_rng(7)
    for _ in range(40):  # JPEG-halo-like speckles, 1-4 px each
        y, x = rng.integers(0, 195), rng.integers(0, 295)
        mask[y : y + rng.integers(1, 3), x : x + rng.integers(1, 3)] = True
    cleaned = remove_small_components(mask, min_area=8)
    assert cleaned[101, 10:290].all(), "curve stroke was removed"
    assert cleaned.sum() < mask.sum(), "no speckle was removed"
    speckle_remaining = cleaned & ~mask[...]  # sanity: nothing invented
    assert not speckle_remaining.any()


# --- color_mask ---


def test_color_mask_catches_exact_and_antialiased_pixels():
    img = _blank()
    cv2.line(img, (50, 150), (350, 60), (220, 30, 30), 2, cv2.LINE_AA)
    mask = color_mask(img, (220, 30, 30), tolerance=25)
    # core pixels along the drawn line must be caught
    xs = np.linspace(60, 340, 30).astype(int)
    ys = np.round(150 + (xs - 50) * (60 - 150) / 300).astype(int)
    hits = np.array([mask[y - 1 : y + 2, x].any() for x, y in zip(xs, ys)])
    assert hits.all()
    # background must not be caught
    assert not mask[250:290, 50:350].any()


def test_color_mask_black_target_catches_jpeg_darkness_range():
    """A black series' pixels vary widely in lightness on JPEG figures but
    stay colorless -- the achromatic dispatch must catch the range while
    still excluding colored and near-white pixels."""
    img = _blank()
    img[100:105, 50:150] = [15, 16, 15]    # dark core
    img[100:105, 150:250] = [80, 82, 80]   # JPEG-lightened stretch
    img[200:205, 50:150] = [220, 30, 30]   # red curve
    mask = color_mask(img, (20, 22, 20), tolerance=25)
    assert mask[102, 60:140].all(), "dark core missed"
    assert mask[102, 160:240].all(), "lightened gray stretch missed"
    assert not mask[202, 60:140].any(), "red pixels caught by black mask"
    assert not mask[250:280, 50:250].any(), "background caught"


def test_color_mask_separates_distinct_colors():
    img = _blank()
    img[100:110, 50:100] = [220, 30, 30]   # red block
    img[200:210, 50:100] = [30, 30, 220]   # blue block
    red_mask = color_mask(img, (220, 30, 30), tolerance=25)
    assert red_mask[100:110, 50:100].all()
    assert not red_mask[200:210, 50:100].any()


def test_hue_recover_catches_desaturated_stroke_not_background_or_neighbour():
    """hue_recover adds a curve's darkened/desaturated pixels (same hue) that
    the distance mask misses, without sweeping in the pale background or a
    different-hued neighbour."""
    img = np.full((20, 20, 3), 245, np.uint8)      # near-white background,
    img[..., 1] = 236; img[..., 2] = 230           # faintly warm-tinted
    img[2:5, 2:5] = (220, 30, 40)                  # bright saturated red
    img[2:5, 8:11] = (120, 92, 86)                 # dark desaturated red (same hue)
    img[10:13, 2:5] = (30, 40, 200)                # blue neighbour

    off = color_mask(img, (220, 30, 40), tolerance=45.0, hue_recover=False)
    on = color_mask(img, (220, 30, 40), tolerance=45.0, hue_recover=True)
    assert off[3, 3] and on[3, 3], "bright core lost"
    assert not off[3, 9] and on[3, 9], "desaturated stroke not recovered by hue"
    assert not on[11, 3], "blue neighbour leaked into the red mask"
    assert not on[0, 0], "pale background swept into the red mask"
    # and red/blue masks stay disjoint under hue recovery
    blue = color_mask(img, (30, 40, 200), tolerance=45.0, hue_recover=True)
    assert blue[11, 3] and not blue[3, 3]
