"""Locked, attempt-based accounting for bounded training jobs."""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked_ledger(path):
    path = Path(path).resolve(strict=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('Budget ledger must be a JSON object')
        yield data


def fields(data):
    if 'formal_limit_iterations' in data:
        return 'formal_limit_iterations', 'formal_used_iterations'
    if 'new_formal_limit' in data:
        return 'new_formal_limit', 'formal_updates_used'
    raise ValueError('Budget ledger has no formal limit')


def reserve(path, attempt_id, run_name, source_sha, iterations):
    with locked_ledger(path) as data:
        limit_key, used_key = fields(data)
        limit, used = data[limit_key], data[used_key]
        if not isinstance(limit, int) or not isinstance(used, int) or used < 0:
            raise ValueError('Invalid budget counters')
        expected = data.get('source_checkpoint_sha256')
        if expected and expected != source_sha:
            raise ValueError('Ledger source checkpoint differs')
        attempts = data.setdefault('attempts', [])
        if any(item['attempt_id'] == attempt_id for item in attempts):
            raise ValueError('Attempt ID already exists')
        reserved = sum(item['reserved_iterations'] for item in attempts
            if item['status'] in ('reserved', 'unknown_consumption'))
        if used + reserved + iterations > limit:
            raise ValueError('Insufficient unreserved training budget')
        attempts.append({'attempt_id':attempt_id, 'run_name':run_name,
            'source_checkpoint_sha256':source_sha, 'reserved_iterations':iterations,
            'status':'reserved', 'actual_update_count':None})
        atomic_json(path, data)


def settle(path, attempt_id, actual_updates, evidence, *, unknown=False):
    with locked_ledger(path) as data:
        _, used_key = fields(data)
        entries = [item for item in data.get('attempts', []) if item['attempt_id'] == attempt_id]
        if len(entries) != 1:
            raise ValueError('Budget attempt identity missing or duplicated')
        entry = entries[0]
        if entry['status'] == 'finalized':
            if entry['actual_update_count'] != actual_updates:
                raise ValueError('Conflicting budget settlement')
            return entry
        if unknown or not isinstance(actual_updates, int) or actual_updates < 0:
            entry.update(status='unknown_consumption', evidence=evidence)
        else:
            if actual_updates > entry['reserved_iterations']:
                raise ValueError('Updates exceed reservation')
            data[used_key] += actual_updates
            entry.update(status='finalized', actual_update_count=actual_updates,
                         evidence=evidence)
        atomic_json(path, data)
        return entry
