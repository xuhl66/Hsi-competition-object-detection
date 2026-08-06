from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from hod26 import audit


def test_candidate_record_is_version_neutral_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = tmp_path / "base.yaml"
    base_config.write_text("value: 1\n", encoding="utf-8")
    candidate_config = tmp_path / "candidate.yaml"
    candidate_config.write_text(
        "__include__:\n  - ./base.yaml\nvalue: 2\n",
        encoding="utf-8",
    )
    source = tmp_path / "model.py"
    source.write_text("MODEL = 'candidate'\n", encoding="utf-8")
    data = tmp_path / "train.json"
    data.write_text('{"images": []}\n', encoding="utf-8")
    parent = tmp_path / "parent.pth"
    parent.write_bytes(b"checkpoint")
    output = tmp_path / "candidate_manifest.json"
    monkeypatch.setattr(
        audit,
        "_git_record",
        lambda: {"head": "abc", "working_tree_dirty": True},
    )
    monkeypatch.setattr(
        audit,
        "_environment_record",
        lambda: {"python": "test"},
    )

    destination = audit.record_candidate(
        candidate="v-next",
        command="python -m train --config candidate.yaml",
        config_paths=[candidate_config],
        source_paths=[source],
        data_paths=[data],
        initialization="continue one parent checkpoint",
        parent_checkpoint=parent,
        output_path=output,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["record_type"] == "candidate_training_stage"
    assert payload["is_final_model"] is False
    assert payload["data_use_declaration"] == {
        "competition_annotations_only_for_task_finetuning": True,
        "declared_external_pretraining": False,
        "external_hyperspectral_dataset_used": False,
        "test_or_ranking_labels_used": False,
    }
    config_names = [
        Path(item["path"]).name
        for item in payload["reconstruction_bundle"]["configs"]
    ]
    assert config_names == ["base.yaml", "candidate.yaml"]
    assert payload["parent_checkpoint"]["sha256"]
    assert payload["reconstruction_bundle_sha256"]
    archive_path = Path(payload["source_snapshot_archive"]["path"])
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert "SNAPSHOT_INDEX.json" in archive.namelist()
        assert any(name.endswith("_model.py") for name in archive.namelist())

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.record_candidate(
            candidate="v-next",
            command="python -m train",
            config_paths=[candidate_config],
            source_paths=[source],
            data_paths=[data],
            initialization="test",
            output_path=output,
        )
