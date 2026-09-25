"""Staged promotion decisions; caller owns counters, windows and command tensors."""


def window_is_due(step, last_check, episodes, commands):
    interval = int(commands.curriculum_check_interval_steps)
    if step - last_check < interval:
        return False
    if episodes < commands.curriculum_min_episodes:
        return False
    return True


def advance_stage(stage, streak, passed, stages, required_passes):
    """A failed window resets the streak; only nonfinal stages can advance."""
    streak = streak + 1 if passed else 0
    advanced = stage < len(stages) - 1 and streak >= int(required_passes)
    if advanced:
        stage += 1
        streak = 0
    return stage, streak, advanced


def log_metrics(stage, streak, stages, last_metrics):
    linear_limit, yaw_limit = stages[stage]
    metrics = {
        'curriculum_stage': float(stage),
        'curriculum_linear_limit': linear_limit,
        'curriculum_yaw_limit': yaw_limit,
        'curriculum_pass_streak': float(streak),
    }
    metrics.update({
        f'curriculum_{name}': float(value)
        for name, value in last_metrics.items()
        if name not in ('passed', 'enough_samples')
    })
    metrics['curriculum_window_passed'] = float(last_metrics['passed'])
    return metrics
