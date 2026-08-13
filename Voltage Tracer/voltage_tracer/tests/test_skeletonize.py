import cv2
import numpy as np

from voltage_tracer.skeletonize import extract_column_candidates, skeletonize_mask


def _thick_curve_mask(h=200, w=300, thickness=3):
    """Boolean mask of a smooth sine-ish curve of known centerline."""
    mask_img = np.zeros((h, w), dtype=np.uint8)
    xs = np.arange(10, w - 10)
    ys = (100 + 50 * np.sin(xs / 40)).astype(int)
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        cv2.line(mask_img, (x0, y0), (x1, y1), 1, thickness)
    return mask_img.astype(bool), dict(zip(xs.tolist(), ys.tolist()))


def test_skeleton_is_thin():
    mask, _ = _thick_curve_mask(thickness=5)
    skel = skeletonize_mask(mask)
    assert skel.sum() < 0.4 * mask.sum()
    # most columns should have exactly 1-2 skeleton pixels
    per_col = skel.sum(axis=0)
    occupied = per_col[per_col > 0]
    assert np.median(occupied) <= 2


def test_candidates_track_known_centerline():
    mask, truth = _thick_curve_mask(thickness=4)
    skel = skeletonize_mask(mask)
    cands = extract_column_candidates(skel)
    errors = []
    for x, true_y in truth.items():
        if x in cands and len(cands[x]) == 1:
            errors.append(abs(cands[x][0] - true_y))
    assert len(errors) > 0.8 * len(truth), "too many missing/split columns"
    assert np.mean(errors) < 2.0, f"centerline error too large: {np.mean(errors):.2f}px"


def test_two_separate_curves_give_two_candidates_per_column():
    mask = np.zeros((200, 100), dtype=bool)
    mask[50, :] = True   # flat line at y=50
    mask[150, :] = True  # flat line at y=150
    cands = extract_column_candidates(skeletonize_mask(mask))
    for x in range(5, 95):
        assert len(cands[x]) == 2
        assert abs(cands[x][0] - 50) <= 2
        assert abs(cands[x][1] - 150) <= 2


def test_merge_gap_merges_adjacent_pixels_only():
    skel = np.zeros((100, 3), dtype=bool)
    # column 0: rows 10 and 12 (gap 2 -> one cluster, mean 11)
    skel[10, 0] = skel[12, 0] = True
    # column 1: rows 10 and 14 (gap 4 -> two clusters)
    skel[10, 1] = skel[14, 1] = True
    cands = extract_column_candidates(skel, merge_gap=2)
    assert cands[0] == [11.0]
    assert cands[1] == [10.0, 14.0]


def test_empty_columns_are_absent_not_zero():
    skel = np.zeros((50, 10), dtype=bool)
    skel[25, 3] = True
    cands = extract_column_candidates(skel)
    assert set(cands.keys()) == {3}


def test_gap_in_curve_leaves_missing_columns_for_tracker():
    """A curve with an occluded stretch must yield absent columns there --
    no interpolated/invented candidates."""
    mask = np.zeros((100, 200), dtype=bool)
    mask[40, 10:80] = True
    mask[40, 120:190] = True  # gap columns 80-119
    cands = extract_column_candidates(skeletonize_mask(mask))
    gap_columns = [x for x in range(85, 115) if x in cands]
    assert gap_columns == []


def test_empty_mask_gives_empty_dict():
    assert extract_column_candidates(np.zeros((50, 50), dtype=bool)) == {}
