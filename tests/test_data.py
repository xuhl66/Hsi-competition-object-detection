from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hod26.data import (
    LetterboxMeta,
    decode_x2cube,
    letterbox_cube,
    parse_voc_annotation,
    restore_boxes,
    make_balanced_folds,
)


def test_x2cube_keeps_official_4x4_band_order() -> None:
    mosaic = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)
    cube = decode_x2cube(mosaic)
    assert cube.shape == (2, 3, 16)
    np.testing.assert_array_equal(cube[0, 0], mosaic[:4, :4].reshape(-1))
    np.testing.assert_array_equal(cube[1, 2], mosaic[4:8, 8:12].reshape(-1))


def test_letterbox_box_roundtrip() -> None:
    cube = np.zeros((100, 200, 16), dtype=np.float32)
    boxes = np.asarray([[10, 20, 150, 80]], dtype=np.float32)
    _, transformed, metadata = letterbox_cube(cube, boxes, 192, 384)
    restored = restore_boxes(torch.from_numpy(transformed), metadata)
    torch.testing.assert_close(restored, torch.from_numpy(boxes))


def test_voc_parser_clips_and_drops_without_mutating_source(tmp_path: Path) -> None:
    annotation = tmp_path / "one.xml"
    annotation.write_text(
        """
<annotation>
  <size><width>999</width><height>999</height><depth>3</depth></size>
  <object><name>apple</name><bndbox>
    <xmin>-2</xmin><ymin>5</ymin><xmax>30</xmax><ymax>40</ymax>
  </bndbox></object>
  <object><name>apple</name><bndbox>
    <xmin>20</xmin><ymin>20</ymin><xmax>20</xmax><ymax>30</ymax>
  </bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )
    original = annotation.read_text(encoding="utf-8")
    objects, audit = parse_voc_annotation(
        annotation, {"apple": 0}, actual_width=100, actual_height=50
    )
    assert objects[0]["bbox"] == [0.0, 5.0, 30.0, 40.0]
    assert len(objects) == 1
    assert audit["clipped_boxes"] == 1
    assert len(audit["dropped"]) == 1
    assert audit["dimension_mismatch"]
    assert annotation.read_text(encoding="utf-8") == original


def test_grouped_balanced_folds_use_every_fold() -> None:
    samples = []
    for index in range(100):
        separated_hash = (index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        samples.append(
            {
                "image_id": str(index),
                "phash": f"{separated_hash:016x}",
                "class_counts": {
                    str(index % 3): 1,
                    str((index + 1) % 3): int(index % 5 == 0),
                },
            }
        )
    folds = make_balanced_folds(samples, num_folds=5, num_classes=3, seed=7)
    assert folds["version"] == 2
    assert sum(folds["fold_image_counts"]) == len(samples)
    assert max(folds["fold_image_counts"]) - min(folds["fold_image_counts"]) <= 2
    assert all(size > 0 for size in folds["fold_image_counts"])
