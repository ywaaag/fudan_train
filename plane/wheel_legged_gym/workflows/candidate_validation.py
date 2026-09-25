"""Additional dynamic gates for a screened candidate, without accepting it."""


def validate_candidate_transitions(candidate, result, stage, round_id, *,
                                   start_stop_enabled, evaluate):
    """Update the existing result in original order; failure propagates to caller.

    A failed static gate cannot be repaired by a successful dynamic gate.
    The historical switches test only runs if preceding gates passed.
    """
    if start_stop_enabled:
        transition = evaluate(candidate, stage, [19, 37, 53],
                              f'r{round_id:02d}_start_stop', transition=True)
        result['passed'] = result['passed'] and transition['passed']
        result['start_stop'] = transition
    if result['passed'] and stage == 'switches':
        transition = evaluate(candidate, stage, [19, 37, 53],
                              f'r{round_id:02d}_transition', transition=True)
        result['passed'] = transition['passed']
        result['transition'] = transition
