"""Process boundary for the separately installed MuJoCo verification repository.

No MuJoCo modules are imported into the Isaac process. The external root is an
explicit argument, read from environment only by the compatibility CLI.
"""
from pathlib import Path
import subprocess


def run_validation(arguments, *, repository, interpreter):
    entry=Path(repository).resolve()/"validate_policy.py"
    if not entry.is_file():
        raise FileNotFoundError("MuJoCo public validation entry missing: "+str(entry))
    return subprocess.run([str(interpreter),str(entry),*arguments]).returncode


def run_mapping_audit(arguments, *, repository, interpreter):
    """Audit captured closed-chain states through the owning repository's CLI."""
    entry = Path(repository).resolve() / 'audit_mapping.py'
    if not entry.is_file():
        raise FileNotFoundError('MuJoCo public mapping audit entry missing: ' + str(entry))
    return subprocess.run([str(interpreter), str(entry), *arguments]).returncode


def run_tree_probe(arguments, *, repository, interpreter):
    """Run the public tree diagnostic without importing or patching its runner."""
    entry = Path(repository).resolve() / 'probe_tree_ramp.py'
    if not entry.is_file():
        raise FileNotFoundError('MuJoCo public tree probe entry missing: ' + str(entry))
    return subprocess.run([str(interpreter), str(entry), *arguments]).returncode
