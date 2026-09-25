"""GUI play command state is session-owned rather than module-global."""
import ast
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'wheel_legged_gym/scripts/play.py'


def test_play_has_explicit_session_state_and_bound_callbacks():
    tree = ast.parse(PATH.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'PlayCommandState')
    fields = {target.attr for node in cls.body if isinstance(node, ast.FunctionDef)
              for stmt in ast.walk(node) if isinstance(stmt, ast.Assign)
              for target in stmt.targets if isinstance(target, ast.Attribute)
              and isinstance(target.value, ast.Name) and target.value.id == 'self'}
    assert {'cmd_x', 'ang_vel', 'cmd_height', 'running', 'turn_left_pressed',
            'turn_right_pressed', 'command_source', 'runtime_limits', 'lock'} <= fields
    for name in ('on_press', 'on_release', 'get_command_state', 'set_panel_command',
                 'apply_manual_commands'):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert fn.args.args[0].arg == 'state', name


def test_play_has_no_module_level_command_state_assignments():
    tree = ast.parse(PATH.read_text())
    forbidden = {'cmd_x', 'ang_vel', 'cmd_height', 'running', 'turn_left_pressed',
                 'turn_right_pressed', 'command_source', 'command_lock', 'runtime_limits'}
    assignments = {target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                   for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                   if isinstance(target, ast.Name)}
    assert not forbidden & assignments
