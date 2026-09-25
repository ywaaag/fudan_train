"""Archive executable package sources without importing simulator modules."""
import hashlib
import json
from pathlib import Path
import shutil


def save_source_snapshot(package_root, run_directory):
    """Copy Python sources by relative path and record their SHA256 hashes.

    Legacy top-level run snapshots remain supported by the application. This
    additional tree preserves implementations behind compatibility wrappers.
    Assets and training outputs are outside package_root and are never copied.
    """
    package_root = Path(package_root)
    destination = Path(run_directory) / "source_snapshot"
    destination.mkdir(exist_ok=False)
    hashes = {}
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "sha256": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
