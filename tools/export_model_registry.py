"""Inventory local model runs and preserve compact comparison evidence in Git."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT/'plane/outputs/policy_comparison_20260919_004138'
OUT = ROOT/'docs/data'


def main():
    manifest = json.loads((JOB/'manifest.json').read_text())
    status = json.loads((JOB/'status.json').read_text())
    assert status['status'] == 'completed' and len(status['completed']) == 48
    selected = {}
    for alias, item in manifest['candidates'].items():
        path = Path(item['checkpoint'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        selected[path.parent.name] = dict(alias=alias, checkpoint=path.name, sha256=item['sha256'])
    runs = []
    for folder in sorted((ROOT/'plane/logs/wheel_legged').iterdir()):
        if not folder.is_dir():
            continue
        source = folder/'policy_experiment.json'
        data = json.loads(source.read_text()) if source.exists() else {}
        iterations = sorted(int(p.stem.split('_')[1]) for p in folder.glob('model_*.pt'))
        runs.append(dict(run=folder.name, path=str(folder.relative_to(ROOT)),
            manifest_present=source.exists(),
            manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
            profile=data.get('name',data.get('profile')), phase=data.get('phase'),
            command_level=data.get('command_level'), randomization_level=data.get('randomization_level'),
            seed=data.get('seed'), resume=data.get('resume'), resume_mode=data.get('resume_mode'),
            source_run=data.get('load_run'), source_iteration=data.get('checkpoint'),
            source_checkpoint=data.get('source_checkpoint'),
            checkpoint_iterations=iterations, compared_checkpoint=selected.get(folder.name)))
    OUT.mkdir(parents=True,exist_ok=True)
    payload = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
        note='File inventory only: highest iteration and file presence do not imply acceptance. Missing manifest fields remain null; source lineage is not guessed.',
        run_count=len(runs), checkpoint_count=sum(len(r['checkpoint_iterations']) for r in runs), runs=runs)
    (OUT/'model_runs_20260919.json').write_text(json.dumps(payload,indent=2)+'\n')
    # Keep small text evidence tracked; models/events/full physics JSON stay in outputs/logs.
    source_csv = JOB/'comparison.csv'
    rows = list(csv.DictReader(source_csv.open()))
    assert len(rows) == 240
    csv_bytes = source_csv.read_bytes().replace(b'\r\n', b'\n')
    (OUT/'policy_comparison_20260919.csv').write_bytes(csv_bytes)
    evidence = dict(job=str(JOB.relative_to(ROOT)), status=status['status'], completed_grids=48,
        csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
        source_csv_sha256=hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        protocol={k:manifest[k] for k in ['schema','seeds','randomization_levels','seconds','warmup',
                  'envs_per_command','initial_commands','higher_stages','gates','selection','scope',
                  'asset_sha256','evaluator_sha256','runner_sha256','git_head','git_status']},
        candidates={alias:dict(checkpoint=str(Path(c['checkpoint']).relative_to(ROOT)),
                    sha256=c['sha256'],source_manifest=c['source_manifest'])
                    for alias,c in manifest['candidates'].items()},
        matching_check=json.loads((JOB/'matching_check.json').read_text()),
        source_hashes=json.loads((JOB/'source_snapshot_sha256.json').read_text()))
    (OUT/'policy_comparison_20260919.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'runs':len(runs),'checkpoints':payload['checkpoint_count'],'comparison_rows':len(rows)}))


if __name__ == '__main__':
    main()
