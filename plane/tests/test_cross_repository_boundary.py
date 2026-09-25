"""Public process interfaces replace cross-checkout imports and runtime patches."""
import ast
import json
from pathlib import Path


def test_declared_mujoco_wrappers_use_only_process_adapter():
    root=Path(__file__).parents[2]
    manifest=json.loads((root/'docs/architecture/process_interfaces.json').read_text())
    assert len(manifest['interfaces'])==3
    for interface in manifest['interfaces']:
        tree=ast.parse((root/interface['training_entry']).read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                imports.append(node.module)
                if node.module=='wheel_legged_gym.adapters.mujoco.api':
                    assert interface['adapter'].rsplit('.',1)[1] in [alias.name for alias in node.names]
        assert 'wheel_legged_gym.adapters.mujoco.api' in imports
        assert not set(imports)&{'mujoco','sim2sim_parity','sim2sim_closed_policy','closed_chain_mapping'}


def test_training_python_never_imports_independent_simulator_modules():
    root=Path(__file__).parents[2]
    prohibited={'sim2sim_parity','sim2sim_closed_policy','closed_chain_mapping','lqr_deploy'}
    violations=[]
    for base in (root/'tools',root/'plane/wheel_legged_gym'):
        for path in base.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                targets=[alias.name for alias in node.names] if isinstance(node,ast.Import) else (
                    [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                if any(target.split('.')[0] in prohibited for target in targets):
                    violations.append(f'{path.relative_to(root)}:{node.lineno}')
    assert violations==[]
