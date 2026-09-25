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
