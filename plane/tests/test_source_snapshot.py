"""Run provenance must retain implementations, not only legacy wrappers."""
import hashlib
import json

import pytest

from wheel_legged_gym.adapters.artifacts.source_snapshot import save_source_snapshot


def test_snapshot_preserves_paths_bytes_and_hashes(tmp_path):
    package = tmp_path / "package"
    (package / "contracts").mkdir(parents=True)
    (package / "legacy.py").write_text("from package.contracts.config import Config\n")
    implementation = package / "contracts/config.py"
    implementation.write_text("class Config:\n    scale = 0.5\n")
    (package / "checkpoint.pt").write_bytes(b"not a source")
    run = tmp_path / "run"
    run.mkdir()
    target = save_source_snapshot(package, run)
    manifest = json.loads((target / "manifest.json").read_text())
    assert set(manifest["sha256"]) == {"legacy.py", "contracts/config.py"}
    for relative, digest in manifest["sha256"].items():
        assert (target / relative).read_bytes() == (package / relative).read_bytes()
        assert digest == hashlib.sha256((target / relative).read_bytes()).hexdigest()
    implementation.write_text("changed after training")
    assert "scale = 0.5" in (target / "contracts/config.py").read_text()
    with pytest.raises(FileExistsError):
        save_source_snapshot(package, run)
