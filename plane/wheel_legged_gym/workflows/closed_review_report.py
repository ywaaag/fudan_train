"""Render candidate closed-review evidence without publishing or promoting it."""


def render_candidate_review(policy, state):
    lines = ['# 候选闭链复核', '', f"策略：{policy}", f"状态：{state['status']}", '',
             '| 分组 | 项目 | 通过 | 失败 |', '|---|---|---|---|']
    for entry in state['reviews']:
        for result in entry['result']['results']:
            passed = result.get('transition_passed', result.get('tracking'))
            lines.append(f"| {entry['name']} | {result['tag']} | {passed} | {result.get('failure')} |")
        if entry['result'].get('skipped'):
            lines += ['', f"跳过（不计通过）：{entry['result']['skipped']}"]
    lines += ['', '不自动提升模型或标记goal完成；与原10000相同输入协议比较后再决策。']
    if state.get('error'):
        lines += ['', state['error']]
    return '\n'.join(lines) + '\n'
