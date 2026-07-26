from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


UPSTREAM = Path("storage/upstream/DEIM")
V2_DATA = Path("storage/v2/dataset/fold0/train.json")


@pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="official pinned DEIM checkout is bootstrapped on the training host",
)
def test_v2_semantic_head_transfer_preserves_unmapped_classes() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import (
        COCO_CLASSES,
        HOD26_CLASSES,
        _semantic_head_tensor,
    )

    source = torch.arange(80, dtype=torch.float32).view(80, 1)
    current = torch.full((18, 1), -1.0)
    mapped = _semantic_head_tensor(
        current,
        source,
        "decoder.enc_score_head.weight",
    )
    assert mapped is not None
    assert mapped[HOD26_CLASSES.index("apple")].item() == COCO_CLASSES.index(
        "apple"
    )
    assert mapped[HOD26_CLASSES.index("people")].item() == COCO_CLASSES.index(
        "person"
    )
    assert mapped[HOD26_CLASSES.index("egg")].item() == -1.0


@pytest.mark.skipif(
    not V2_DATA.is_file(),
    reason="V2 COCO views are generated from the official local dataset",
)
def test_v2_coco_view_is_complete_and_0_based() -> None:
    payload = json.loads(V2_DATA.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 2400
    assert [item["id"] for item in payload["categories"]] == list(range(18))
    assert all(
        annotation["bbox"][2] > 0 and annotation["bbox"][3] > 0
        for annotation in payload["annotations"]
    )
