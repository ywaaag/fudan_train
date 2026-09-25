"""Architecture tests run without importing either simulator."""
import importlib.util
from pathlib import Path


def test_source_import_graph_is_acyclic():
    path=Path(__file__).resolve().parents[2]/'tools/check_architecture.py'
    spec=importlib.util.spec_from_file_location('architecture_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.inventory()
    assert report['cycles']==[]
    assert report['layer_violations']==[]
    assert report['simulator_import_violations']==[]
    assert report['missing_local_imports']==[]
    assert module.forbidden_edge('wheel_legged_gym.evaluation.metrics','wheel_legged_gym.app.cli')
    assert module.forbidden_edge('wheel_legged_gym.domain.commands','wheel_legged_gym.learning.ppo')
    assert not module.forbidden_edge('wheel_legged_gym.workflows.job','wheel_legged_gym.evaluation.metrics')
    assert module.forbidden_edge('wheel_legged_gym.envs.base.legged_robot','wheel_legged_gym.app.bootstrap')
    assert module.forbidden_edge('wheel_legged_gym.utils.math','wheel_legged_gym.envs.base.legged_robot')
    assert module.forbidden_edge('wheel_legged_gym.app.bootstrap','wheel_legged_gym.scripts.train')
    assert not module.forbidden_edge('wheel_legged_gym.envs.base.legged_robot','wheel_legged_gym.adapters.isaacgym.base_task')
    assert module.forbidden_edge('wheel_legged_gym.app.cli','tools.run_motion_goal')
    assert module.forbidden_edge('tools.run_example','tools.run_motion_goal')
    assert module.forbidden_edge('wheel_legged_gym.adapters.export','export_onnx.export_onnx')
    assert module.forbidden_edge('wheel_legged_gym.scripts.train','wheel_legged_gym.scripts.play')
    assert not module.forbidden_edge('tools.run_motion_goal','wheel_legged_gym.app.motion_supervisor')


def test_package_initializers_do_not_register_tasks():
    import ast
    root=Path(__file__).resolve().parents[1]/'wheel_legged_gym'
    for name in ['envs','utils','learning/modules']:
        tree=ast.parse((root/name/'__init__.py').read_text())
        assert not any(isinstance(n,ast.Call) for n in ast.walk(tree))


def test_package_uses_explicit_import_names():
    import ast
    root=Path(__file__).parents[1]/'wheel_legged_gym'
    violations=[]
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and any(alias.name=='*' for alias in node.names):
                violations.append(f'{path.relative_to(root)}:{node.lineno}')
    assert violations==[]


def test_production_modules_do_not_evaluate_source_strings():
    import ast
    root = Path(__file__).resolve().parents[2]
    violations = []
    for base in (root/'plane/wheel_legged_gym', root/'plane/export_onnx', root/'tools'):
        for path in base.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in {'eval', 'exec', '__import__'}):
                    violations.append(f'{path.relative_to(root)}:{node.lineno}')
    assert violations == []


def test_no_silently_overridden_class_methods():
    """Duplicate methods hide implementations; property accessors are explicit exceptions."""
    import ast
    root = Path(__file__).resolve().parents[1] / 'wheel_legged_gym'
    duplicates = []
    for path in root.rglob('*.py'):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            names = set()
            for node in cls.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                accessor = any(isinstance(d, ast.Attribute) and d.attr in {'setter', 'deleter'}
                               and isinstance(d.value, ast.Name) and d.value.id == node.name
                               for d in node.decorator_list)
                if node.name in names and not accessor:
                    duplicates.append(f'{path.relative_to(root)}:{node.lineno} {cls.name}.{node.name}')
                names.add(node.name)
    assert duplicates == []


def test_removed_utility_facades_are_not_imported():
    import ast
    root=Path(__file__).resolve().parents[2]
    removed={'wheel_legged_gym.utils.helpers','wheel_legged_gym.utils.task_registry',
             'wheel_legged_gym.utils.terrain'}
    removed.update('wheel_legged_gym.envs.base.'+name for name in (
        'command_sampling','command_curriculum','height_commands','reward_terms',
        'start_stop_commands','base_task','base_config','legged_robot_config'))
    for base in (root/'plane/wheel_legged_gym',root/'plane/tests',root/'tools'):
        for path in base.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names=[item.name for item in node.names] if isinstance(node,ast.Import) else (
                    [node.module] if isinstance(node,ast.ImportFrom) else [])
                if isinstance(node,ast.ImportFrom) and node.level and root/'plane' in path.parents:
                    parts=list(path.relative_to(root/'plane').with_suffix('').parts)
                    package=parts[:-1]
                    names=['.'.join(package[:len(package)-node.level+1]+([node.module] if node.module else []))]
                assert not removed.intersection(names),str(path)


def test_missing_absolute_and_function_local_relative_imports_are_reported(tmp_path):
    path=Path(__file__).resolve().parents[2]/'tools/check_architecture.py'
    spec=importlib.util.spec_from_file_location('missing_import_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=tmp_path
    package=tmp_path/'plane/wheel_legged_gym/domain'
    package.mkdir(parents=True)
    (package/'present.py').write_text('VALUE = 1\n')
    (package/'consumer.py').write_text(
        'from wheel_legged_gym.domain.present import VALUE\n'
        'from wheel_legged_gym.domain import present\n'
        'import wheel_legged_gym.deleted\n'
        'def run():\n    from .missing import value\n')
    report=module.inventory()
    assert report['missing_local_imports']==[
        {'source':'wheel_legged_gym.domain.consumer','target':'wheel_legged_gym.deleted','line':3},
        {'source':'wheel_legged_gym.domain.consumer','target':'wheel_legged_gym.domain.missing','line':5},
    ]
