"""Render a motion job's recorded state without I/O or notifications."""
import json


def render_progress(state):
    lines = ['# Autonomous motion goal progress', '', f"Status: {state['status']}",
             f"Stage: {state['stage']}; round: {state['round']}",
             f"Accepted checkpoint: `{state['accepted']}`", '',
             'Goal remains incomplete until final motion and independent sim2sim review.', '',
             'Each round: 500 iterations, full-state resume, fixed recipe, 3-seed gates.', '',
             '## Decisions', '']
    lines += [json.dumps(h, ensure_ascii=False) for h in state['history']]
    return '\n'.join(lines) + '\n'
