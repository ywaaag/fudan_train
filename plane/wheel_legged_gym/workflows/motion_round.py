"""Advance one evaluated motion round; application owns persistence and processes."""
from dataclasses import dataclass
from pathlib import Path

from wheel_legged_gym.workflows.candidate_decision import decide_candidate, recovery_focus
from wheel_legged_gym.workflows.candidate_validation import validate_candidate_transitions


@dataclass(frozen=True)
class RoundProgress:
    source: Path
    stage_index: int
    baseline: dict
    stagnant: int
    focus: list
    curriculum_finished: bool = False


@dataclass(frozen=True)
class RoundMode:
    start_stop_enabled: bool
    recover_motion: bool
    turn_curriculum: bool
    speed_envelope: bool
    basic_motion: bool


def advance_round(candidates, progress: RoundProgress, *, stages, round_id: int,
                  mode: RoundMode, state, evaluate, export) -> RoundProgress:
    """Record decisions in state, preserving publication and failure ordering.

    state is the caller-owned status journal, not a global registry. Export must
    succeed before accepted/history change. Enter-stage evaluation happens after
    that change, so its failure retains the published acceptance in the journal.
    The caller saves state and handles stop limits; this function never does I/O
    except through the two explicit callbacks.
    """
    source = progress.source
    stage_index = progress.stage_index
    baseline = progress.baseline
    stagnant = progress.stagnant
    focus = progress.focus
    stage = stages[stage_index]
    if not candidates:
        state['history'].append({
            'round': round_id, 'stage': stage,
            'decision': 'no candidate passed safety and posture retention; retained source',
        })
        stagnant += 1
    else:
        _, candidate, result = min(candidates)
        validate_candidate_transitions(
            candidate, result, stage, round_id,
            start_stop_enabled=mode.start_stop_enabled, evaluate=evaluate,
        )
        decision = decide_candidate(result, baseline)
        if decision.accepted:
            export(candidate, f'accepted_{stage}')
            state['accepted'] = str(candidate)
            state['history'].append({
                'round': round_id, 'stage': stage, 'decision': 'stage accepted',
                'checkpoint': str(candidate), 'result': result,
            })
            source = candidate
            stage_index += 1
            focus = []
            stagnant = 0
            if stage_index == len(stages):
                state['status'] = (
                    'motion_and_geometry_passed_pending_dynamic_validation' if mode.recover_motion else
                    'turn_grid_passed_pending_transitions_and_sim2sim' if mode.turn_curriculum else
                    'simulation_curriculum_passed_pending_sim2sim_and_smoothness_review'
                )
                return RoundProgress(source, stage_index, baseline, stagnant, focus, True)
            baseline = evaluate(source, stages[stage_index], [19, 37, 53],
                                f'enter_{stages[stage_index]}')
        else:
            state['history'].append({
                'round': round_id, 'stage': stage, 'decision': decision.history_label,
                'checkpoint': str(candidate), 'result': result,
            })
            if decision.improved:
                source = candidate
                baseline = result
                stagnant = 0
            else:
                stagnant += 1
            # Sampling weights may change; every stage anchor remains present.
            if not mode.recover_motion or mode.speed_envelope or mode.basic_motion:
                focus = recovery_focus(baseline, stagnant)
    return RoundProgress(source, stage_index, baseline, stagnant, focus)
