"""Summary boundaries retain Python slicing, tag order and rounding behavior."""
import copy
import pytest
from wheel_legged_gym.evaluation.training_summary import (
    METRIC_KEYS, summarize_scalars, through_iteration,
)


@pytest.mark.parametrize('window', [1, 2, 50, 0, -1])
def test_summary_preserves_tail_semantics(window):
    data = {METRIC_KEYS[0]: [(1, .123456789), (2, 2.25), (4, -.25)],
            METRIC_KEYS[-1]: [(3, .5), (6, .75)], 'Other/tag': [(1, 99.)]}
    before = copy.deepcopy(data)
    result = summarize_scalars('relative/run', data, window)
    assert result['run'] == 'relative/run'
    assert list(result['metrics']) == [METRIC_KEYS[0], METRIC_KEYS[-1]]
    for key, summary in result['metrics'].items():
        values = data[key][-window:]
        assert summary == {'step': values[-1][0], 'last': round(values[-1][1], 6),
                           'tail_mean': round(sum(v for _, v in values)/len(values), 6)}
    assert data == before


def test_missing_tags_are_omitted_and_empty_existing_tag_still_fails():
    assert summarize_scalars('run', {}) == {'run': 'run', 'metrics': {}}
    with pytest.raises(IndexError):
        summarize_scalars('run', {METRIC_KEYS[0]: []})


def test_checkpoint_cutoff_excludes_later_events_without_changing_input():
    data = {METRIC_KEYS[0]: [(11248,1.),(11249,2.),(11250,99.)],
            METRIC_KEYS[1]: [(11251,5.)]}
    before = copy.deepcopy(data)
    filtered = through_iteration(data,11250)
    assert list(filtered) == [METRIC_KEYS[0]]
    assert summarize_scalars('run',filtered,50)['metrics'][METRIC_KEYS[0]] == {
        'step':11249,'last':2.,'tail_mean':1.5}
    assert data == before
