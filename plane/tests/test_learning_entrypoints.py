"""Learning has one canonical implementation namespace, without forwarding packages."""
import ast
from pathlib import Path

from wheel_legged_gym.learning.modules.api import ActorCriticSequence
from wheel_legged_gym.learning.modules.actor_critic_sequence import ActorCriticSequence as Sequence


def test_network_api_is_canonical():
    assert ActorCriticSequence is Sequence


def test_production_code_does_not_import_removed_learning_namespace():
    root=Path(__file__).parents[2]
    assert not list((root/'plane/wheel_legged_gym/rsl_rl').rglob('*.py'))
    for base in (root/'plane/wheel_legged_gym',root/'plane/export_onnx',root/'tools'):
        for path in base.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                modules=[a.name for a in node.names] if isinstance(node,ast.Import) else (
                    [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                assert not any(m.startswith('wheel_legged_gym.rsl_rl') or m=='rsl_rl'
                               for m in modules),path
