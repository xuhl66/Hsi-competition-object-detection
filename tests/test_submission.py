from __future__ import annotations

import csv

from hod26.submission import validate_submission, write_submission


def test_submission_has_rule_required_id_and_valid_cube_coordinates(tmp_path) -> None:
    samples = [{"image_id": "1000", "width": 512, "height": 256}]
    rows = [
        {
            "image_id": "1000",
            "class_id": 17,
            "confidence": 0.75,
            "x1": 1,
            "y1": 2,
            "x2": 30,
            "y2": 40,
        }
    ]
    output = write_submission(rows, tmp_path / "submission.csv", include_id=True)
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert next(reader) == [
            "id",
            "image_id",
            "class_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        assert next(reader)[0] == "0"
    report = validate_submission(output, samples, 18, include_id=True)
    assert report["valid"], report["errors"]

