"""Classify a validated candidate without changing source or publishing artifacts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    improved: bool
    history_label: str


def decide_candidate(result, baseline):
    # Preserve the strict margin and its evaluation before the passed check.
    improved = result['score'] < baseline['score'] - .01
    if result['passed']:
        return CandidateDecision(True, improved, 'stage accepted')
    return CandidateDecision(False, improved,
                             'improved candidate' if improved else 'rollback retained source')


def recovery_focus(baseline, stagnant):
    """Retain original failed-command order and repetition cap; no command removal."""
    return [list(pair) for pair in baseline['failed_commands']] * min(4, stagnant + 2)
