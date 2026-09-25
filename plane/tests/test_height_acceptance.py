"""Additional height checks retain strict boundaries and base rejection reasons."""
import pytest
from wheel_legged_gym.evaluation import height_acceptance


@pytest.mark.parametrize('stage,tolerance', [('micro', .005), ('fixed35', .015), ('dual', .015)])
@pytest.mark.parametrize('offset', [-1e-9, 0., 1e-9])
@pytest.mark.parametrize('contact', [0., .001])
@pytest.mark.parametrize('base_passed', [False, True])
def test_additional_gates(monkeypatch, stage, tolerance, offset, contact, base_passed):
    base = {'passed':base_passed, 'failed_checks':[] if base_passed else ['base_gate']}
    monkeypatch.setattr(height_acceptance, 'gate', lambda row:base)
    row = {'metrics':{'height_mae':tolerance+offset},
           'nonwheel_contact_full_fraction':contact, 'gate':{'stale':True}}
    result = height_acceptance.assess_height_row(row, stage)
    expected = ([] if base_passed else ['base_gate'])
    if offset > 0: expected.append('height_tracking')
    if contact > 0: expected.append('full_contact')
    assert result is row['gate'] is base
    assert result['failed_checks'] == expected
    assert result['passed'] == (base_passed and offset <= 0 and contact == 0)
