import hashlib
from zipfile import ZipFile

import pytest

from experiments.word_games import import_bundle as bundle


def test_fresh_data_import_accepts_identical_already_vendored_source(tmp_path, monkeypatch):
    archive = tmp_path / "bundle.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(bundle.V2 + "/bridge.py", "# existing vendor\n")
    target = tmp_path / "vendor"
    target.mkdir()
    (target / "bridge.py").write_text("# existing vendor\n")
    monkeypatch.setattr(bundle, "ARCHIVE", archive)
    monkeypatch.setattr(bundle, "ARCHIVE_SHA", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(bundle, "INPUTS", tmp_path / "data")
    monkeypatch.setattr(bundle, "VENDOR", target)
    monkeypatch.setattr(bundle, "SELECT", ("bridge.py",))
    assert bundle.run()["imported_files"] == 1
    assert bundle.run()["archive_sha256"] == bundle.ARCHIVE_SHA


def test_changed_vendor_is_not_overwritten(tmp_path, monkeypatch):
    archive = tmp_path / "bundle.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(bundle.V2 + "/bridge.py", "# supplied vendor\n")
    target = tmp_path / "vendor"
    target.mkdir()
    (target / "bridge.py").write_text("# user modification\n")
    monkeypatch.setattr(bundle, "ARCHIVE", archive)
    monkeypatch.setattr(bundle, "ARCHIVE_SHA", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(bundle, "INPUTS", tmp_path / "data")
    monkeypatch.setattr(bundle, "VENDOR", target)
    monkeypatch.setattr(bundle, "SELECT", ("bridge.py",))
    with pytest.raises(ValueError):
        bundle.run()
    assert (target / "bridge.py").read_text() == "# user modification\n"
    assert not bundle.INPUTS.exists()
