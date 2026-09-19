"""Compare the read-only upstream Fudan source contract with this port."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=Path('/home/kellen/fudan_rl_wheel_leg/plane')
FILES={
 'wheel_config':'wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py',
 'base_config':'wheel_legged_gym/envs/base/legged_robot_config.py',
 'base_env':'wheel_legged_gym/envs/base/legged_robot.py',
 'ppo':'wheel_legged_gym/rsl_rl/algorithms/ppo.py',
 'sequence_policy':'wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def literals(path):
    out={}
    if not path.exists(): return out
    def walk(body,prefix=''):
        for n in body:
            if isinstance(n,ast.ClassDef): walk(n.body,prefix+n.name+'.')
            elif isinstance(n,ast.Assign):
                for t in n.targets:
                    if isinstance(t,ast.Name):
                        try: out[prefix+t.id]=ast.literal_eval(n.value)
                        except (ValueError,TypeError): pass
    walk(ast.parse(path.read_text()).body)
    return out

def main():
    report={'upstream':str(UP),'port':str(ROOT),'read_only':True,
            'status':'non_equivalent_physics_and_training_contract','files':{}}
    for key,rel in FILES.items():
        a,b=UP/rel,ROOT/'plane'/rel
        old,new=literals(a),literals(b)
        changed={k:{'upstream':v,'port':new.get(k)} for k,v in old.items() if new.get(k)!=v}
        diff=''.join(difflib.unified_diff(
            a.read_text().splitlines(True) if a.exists() else [],
            b.read_text().splitlines(True) if b.exists() else [],
            fromfile=str(a),tofile=str(b),n=1))
        report['files'][key]={'relative_path':rel,'upstream_sha256':digest(a),
            'port_sha256':digest(b),'literal_config_changes':changed,
            'diff_lines':len(diff.splitlines()),'diff_excerpt':diff.splitlines()[:80]}
    report['explicit_findings']=[
        {'area':'asset','upstream':'resources/robots/infantry_V4/urdf/infantry_V4_increase.urdf',
         'port':'assets/wheel_leg_train.urdf','implication':'different inertial/contact/joint model'},
        {'area':'initial_state','upstream':'base z=0.1; leg angles +0.2/+0.4 and -0.2/-0.4',
         'port':'base z=0.4; all leg angles 0','implication':'different equilibrium and reset distribution'},
        {'area':'wheel_pd','upstream':'wheel damping=0.2','port':'wheel damping=1.0',
         'implication':'different wheel velocity tracking and torque response'},
        {'area':'reward','upstream':'tracking_lin_vel=exp(-error^2/sigma); tracking_sigma=0.25',
         'port':'normalized_v1 coarse/fine/Huber gap plus safety/contact/gates',
         'implication':'reward value and gradient have different semantics'},
        {'area':'optimizer','upstream':'default actor/encoder lr=1e-3; entropy=.01',
         'port':'active migration lr=1e-5; entropy=.001','implication':'optimization timescale differs'},
        {'area':'policy_update','upstream':'encoder extra optimizer only',
         'port':'encoder extra optimizer plus symmetry estimator loss',
         'implication':'actor effective input can move after PPO actor update'},
    ]
    out=ROOT/'docs/data/source_contract_comparison_20260919.json'
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'out':str(out),'findings':len(report['explicit_findings'])}))

if __name__=='__main__': main()
