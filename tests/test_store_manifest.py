import json
from pathlib import Path

import pytest

from deploy.store_manifest import FILES, NOTICES, create, verify


def test_transfer_manifest_detects_tampering_and_rejects_path_escape(tmp_path):
    store, notices, target = [tmp_path / name for name in ("source", "notices", "target")]
    for directory in (store, notices, target):
        directory.mkdir()
    for directory, names in ((store, FILES), (notices, NOTICES)):
        for name in names:
            (directory / name).write_bytes(name.encode())
            (target / name).write_bytes(name.encode())
    manifest = create(store, notices)
    assert str(tmp_path) not in json.dumps(manifest)
    verify(target, manifest)
    (target / "experts.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify(target, manifest)
    manifest["files"][0]["name"] = "../outside"
    with pytest.raises(ValueError, match="expected"):
        verify(target, manifest)


def test_notices_include_upstream_license_and_changes():
    notices = Path(__file__).resolve().parents[1] / "deploy" / "model-notices"
    assert all((notices / name).is_file() for name in NOTICES)
    assert "Tongyi Qianwen LICENSE AGREEMENT" in (notices / "LICENSE").read_text()
    assert "ExpertRelay" in (notices / "CHANGES").read_text()
