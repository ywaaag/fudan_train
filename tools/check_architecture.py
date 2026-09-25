"""Inventory local dependencies without importing simulators (including local imports and package initialization)."""
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

# Enforced for migrated modules. Legacy tools/scripts remain explicit migration work,
# rather than a broad allowlist that could silently weaken these boundaries.
ALLOWED_LAYERS={
    'contracts': {'contracts'},
    'domain': {'contracts','domain'},
    'experiments': {'contracts','domain','experiments'},
    'evaluation': {'contracts','domain','evaluation'},
    'learning': {'contracts','domain','learning'},
    'ports': {'contracts','ports'},
    'workflows': {'contracts','domain','evaluation','learning','ports','workflows'},
    'adapters': {'contracts','domain','evaluation','learning','ports','adapters'},
    # Environment orchestration may use simulator adapters, never app launchers.
    'envs': {'contracts','domain','adapters','envs'},
    'utils': {'utils'},
    'app': {'contracts','domain','experiments','evaluation','learning','ports',
            'workflows','adapters','envs','utils','app'},
}


def layer(module):
    parts=module.split('.')
    return parts[1] if len(parts)>1 and parts[0]=='wheel_legged_gym' else None


def forbidden_edge(source,target):
    # Executable entrypoints are leaves of the import graph, not reusable APIs.
    # Their reuse must be a declared process invocation, never a Python import.
    entrypoints = ('tools.', 'export_onnx.', 'wheel_legged_gym.scripts.')
    if target.startswith(entrypoints):
        return True
    source_layer=layer(source);target_layer=layer(target)
    return bool(source_layer in ALLOWED_LAYERS and target_layer is not None
                and target_layer not in ALLOWED_LAYERS[source_layer])


def inventory():
    modules={}
    for base,prefix in [(ROOT/'plane/wheel_legged_gym','wheel_legged_gym'),(ROOT/'tools','tools'),(ROOT/'plane/export_onnx','export_onnx')]:
        for path in sorted(base.rglob('*.py')):
            parts=list(path.relative_to(base).with_suffix('').parts)
            if parts[-1]=='__init__':parts.pop()
            modules['.'.join([prefix]+parts)]=path
    edges=set()
    missing_imports=[]
    # Namespace packages can have no __init__.py; they are still valid imports.
    local_names=set(modules)
    for module in modules:
        parts=module.split('.')
        local_names.update('.'.join(parts[:i]) for i in range(1,len(parts)))
    for name,path in modules.items():
        package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            targets=[]
            if isinstance(node,ast.Import):targets=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                stem=node.module or ''
                if node.level:stem='.'.join(package.split('.')[:len(package.split('.'))-node.level+1]+([stem] if stem else []))
                targets=[stem]+[stem+'.'+a.name for a in node.names]
            requested=([a.name for a in node.names] if isinstance(node,ast.Import)
                       else [stem] if isinstance(node,ast.ImportFrom) else [])
            for target in requested:
                if target.startswith('wheel_legged_gym.') and target not in local_names:
                    missing_imports.append({'source':name,'target':target,'line':node.lineno})
            for target in targets:
                if target not in modules and 'tools.'+target in modules:target='tools.'+target
                if target not in modules:continue
                edges.add((name,target,node.lineno,'import'))
                for i in range(1,len(target.split('.'))):
                    parent='.'.join(target.split('.')[:i])
                    if parent in modules and parent!=name:edges.add((name,parent,node.lineno,'package_init'))
    graph={m:set() for m in modules}
    for a,b,_,_ in edges:graph[a].add(b)
    index=0;indices={};low={};stack=[];active=set();cycles=[]
    def visit(v):
        nonlocal index
        indices[v]=low[v]=index;index+=1;stack.append(v);active.add(v)
        for w in sorted(graph[v]):
            if w not in indices:visit(w);low[v]=min(low[v],low[w])
            elif w in active:low[v]=min(low[v],indices[w])
        if low[v]==indices[v]:
            group=[]
            while True:
                w=stack.pop();active.remove(w);group.append(w)
                if w==v:break
            if len(group)>1 or v in graph[v]:cycles.append(sorted(group))
    for name in sorted(graph):
        if name not in indices:visit(name)
    violations=[{'source':a,'target':b,'line':n} for a,b,n,k in sorted(edges)
                if k=='import' and forbidden_edge(a,b)]
    simulator_imports=[]
    for name,path in modules.items():
        if layer(name) not in {'contracts','domain','evaluation','learning','ports','workflows'}:continue
        for node in ast.walk(ast.parse(path.read_text())):
            imports=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for imported in imports:
                if imported.split('.')[0] in {'isaacgym','mujoco','sim2sim_closed_policy'}:
                    simulator_imports.append({'source':name,'target':imported,'line':node.lineno})
    return {'modules':{k:str(v.relative_to(ROOT)) for k,v in sorted(modules.items())},
            'edges':[{'source':a,'target':b,'line':n,'kind':k} for a,b,n,k in sorted(edges)],
            'cycles':cycles,'layer_violations':violations,'simulator_import_violations':simulator_imports,
            'missing_local_imports':missing_imports,
            'scope':'Local imports, including function-local imports and package init. Layer enforcement applies to ALLOWED_LAYERS; legacy tools/scripts migration remains separate. Runtime process dependencies require separate review.'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--write',type=Path);a=p.parse_args();result=inventory()
    if a.write:a.write.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'modules':len(result['modules']),'edges':len(result['edges']),
        **{k:result[k] for k in ['cycles','layer_violations','simulator_import_violations','missing_local_imports']}},indent=2))
    return bool(result['cycles'] or result['layer_violations'] or result['simulator_import_violations'] or result['missing_local_imports'])


if __name__=='__main__':raise SystemExit(main())
