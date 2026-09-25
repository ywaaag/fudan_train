"""Two-pass checkpoint screening; callbacks own simulation execution."""


def screen_candidates(folder, source_iteration, stage, round_id, *, evaluate):
    """Keep original +100..+500 screening and top-two three-seed validation.

    Sorting remains tuple-based: equal scores are ordered by checkpoint path.
    Passing safety alone never bypasses an explicitly failed posture check.
    """
    screened = []
    for iteration in range(source_iteration + 100, source_iteration + 501, 100):
        checkpoint = folder / f'model_{iteration}.pt'
        score = evaluate(checkpoint, stage, [19], f'r{round_id:02d}_m{iteration}')
        if score['safe'] and score.get('posture_retained', True):
            screened.append((score['score'], checkpoint))
    candidates = []
    for _, checkpoint in sorted(screened)[:2]:
        score = evaluate(checkpoint, stage, [19, 37, 53],
                         f'r{round_id:02d}_m{checkpoint.stem.split("_")[-1]}')
        if score['safe'] and score.get('posture_retained', True):
            candidates.append((score['score'], checkpoint, score))
    return candidates
