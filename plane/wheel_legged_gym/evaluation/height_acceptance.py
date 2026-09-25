"""Height-course checks layered on the existing motion gate."""
from wheel_legged_gym.evaluation.gates import gate


def assess_height_row(row, stage):
    """Attach the historical gate to this row, retaining failed-check order."""
    row['gate'] = gate(row)
    tolerance = .005 if stage == 'micro' else .015
    if row['metrics']['height_mae'] > tolerance:
        row['gate']['passed'] = False
        row['gate']['failed_checks'].append('height_tracking')
    if row['nonwheel_contact_full_fraction'] > 0:
        row['gate']['passed'] = False
        row['gate']['failed_checks'].append('full_contact')
    return row['gate']


def assess_height_transitions(records, responses):
    """Return separate steady/response gates without mutating evaluation records.

    Preserve the dual-height protocol's <= comparisons (including NaN rejection),
    rather than substituting the course's > rejection checks above.
    """
    passed = all(
        gate(row)['passed'] and row['metrics']['height_mae'] <= .015
        and row['nonwheel_contact_full_fraction'] == 0 for row in records
    )
    settled = all(
        axis['settled_envs'] == axis['total_envs']
        and axis['worst_settling_s'] is not None
        and axis['worst_settling_s'] <= 3
        for report in responses for row in report['rows'] for axis in row['response'].values()
    )
    return passed, settled
