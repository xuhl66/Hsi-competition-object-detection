from __future__ import annotations

import numpy as np

from hod26.v6.calibration import (
    NATIVE_MAX_PER_IMAGE,
    NATIVE_PRE_NMS_TOPK,
    NUM_CLASSES,
    _fixed_candidate_invariants,
    _restore_query_boxes,
    decode_calibrated,
    decode_cached_native,
    decode_fixed_selected_margin,
    margin_calibrated_probabilities,
    match_native_queries,
    native_query_purities,
    prediction_counts,
    quantile_factors,
)


def test_restore_query_boxes_maps_normalized_cxcywh_to_original() -> None:
    boxes = _restore_query_boxes(
        np.asarray([[0.5, 0.5, 0.2, 0.4]], np.float32),
        resized_width=1000,
        resized_height=500,
        scale_x=2.0,
        scale_y=2.0,
        original_width=500,
        original_height=250,
    )
    np.testing.assert_allclose(boxes, [[200, 75, 300, 175]], atol=1e-5)


def test_calibration_changes_only_scores_and_cross_class_selection() -> None:
    pool = np.zeros((1, NUM_CLASSES, 2, 5), np.float32)
    counts = np.zeros((1, NUM_CLASSES), np.int16)
    pool[0, 0, 0] = [0, 0, 10, 10, 0.8]
    pool[0, 1, 0] = [1, 1, 11, 11, 0.7]
    counts[0, :2] = 1
    factors = np.ones(NUM_CLASSES, np.float64)
    factors[0] = 0.5
    result = decode_calibrated(pool, counts, factors=factors, caps=2, max_per_image=1)
    assert len(result[0][0]) == 0
    np.testing.assert_allclose(result[0][1][0], [1, 1, 11, 11, 0.7])
    assert prediction_counts(result)[:2] == [0, 1]


def test_quantile_factors_are_bounded_and_downweight_high_scale_class() -> None:
    scores = np.ones((2, 3, NUM_CLASSES), np.float32) * 0.1
    scores[:, :, 5] = 0.8
    factors, anchors = quantile_factors(scores, quantile=0.9, exponent=0.5)
    assert factors.shape == anchors.shape == (NUM_CLASSES,)
    assert np.all(factors > 0)
    assert np.all(factors <= 1)
    assert factors[5] < factors[0]


def test_margin_calibration_suppresses_ambiguous_class_more() -> None:
    import torch

    probabilities = torch.tensor([[[0.9, 0.05] + [0.01] * 16, [0.2, 0.2] + [0.01] * 16]])
    calibrated = margin_calibrated_probabilities(probabilities, [1.0] * NUM_CLASSES)
    dominant_ratio = calibrated[0, 0, 0] / probabilities[0, 0, 0]
    ambiguous_ratio = calibrated[0, 1, 0] / probabilities[0, 1, 0]
    assert dominant_ratio > ambiguous_ratio
    assert torch.all(calibrated <= probabilities)


def _toy_fixed_cache() -> dict[str, np.ndarray]:
    scores = np.full((1, NATIVE_PRE_NMS_TOPK, NUM_CLASSES), 1e-4, np.float32)
    boxes = np.zeros((1, NATIVE_PRE_NMS_TOPK, 4), np.float32)
    boxes[:, :, 2:] = 1
    boxes[0, 7] = [10, 20, 30, 40]
    boxes[0, 11] = [50, 60, 80, 90]
    scores[0, 7, 0] = 0.90
    scores[0, 7, 1] = 0.60
    scores[0, 11, 1] = 0.80
    scores[0, 11, 0] = 0.05
    native = np.zeros((1, NATIVE_MAX_PER_IMAGE, 6), np.float32)
    native[0, 0] = [0, 10.1, 19.9, 30.1, 39.9, 0.90]
    native[0, 1] = [1, 49.9, 60.1, 79.9, 90.1, 0.80]
    return {
        "scores": scores,
        "boxes": boxes,
        "native": native,
        "native_counts": np.asarray([2], np.int16),
        "image_ids": np.asarray([0], np.int64),
        "widths": np.asarray([100], np.int32),
        "heights": np.asarray([100], np.int32),
    }


def test_fixed_candidate_margin_preserves_boxes_labels_and_counts() -> None:
    cache = _toy_fixed_cache()
    query_indexes, report = match_native_queries(cache)
    assert query_indexes[0, :2].tolist() == [7, 11]
    assert report["max_box_error_px"] < 0.2
    purities = native_query_purities(cache, query_indexes)
    native = decode_cached_native(cache)
    zero = decode_fixed_selected_margin(
        cache,
        [0.0] * NUM_CLASSES,
        query_indexes=query_indexes,
        purities=purities,
    )
    assert _fixed_candidate_invariants(native, zero)["changed_scores"] == 0

    exponents = [0.0] * NUM_CLASSES
    exponents[0] = 1.0
    calibrated = decode_fixed_selected_margin(
        cache,
        exponents,
        query_indexes=query_indexes,
        purities=purities,
    )
    invariants = _fixed_candidate_invariants(native, calibrated)
    assert invariants["labels_and_class_counts_identical"]
    assert invariants["changed_scores"] == 1
    assert calibrated[0][0][0, 4] < native[0][0][0, 4]
    np.testing.assert_array_equal(calibrated[0][0][:, :4], native[0][0][:, :4])
