"""One real PPO update, diagnostics only; never writes a trained checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plane'))
import isaacgym
import torch
from wheel_legged_gym.app.bootstrap import create_task_registry
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment
from wheel_legged_gym.adapters.isaacgym.evaluation_setup import build_evaluation_args


def main():
    task_registry = create_task_registry()
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cfg,train=task_registry.get_cfgs('wheel_legged')
    apply_policy_experiment(cfg,'HEIGHT_COURSE',train)
    args=build_evaluation_args();args.num_envs=512;args.seed=23;cfg.env.num_envs=512
    env,_=task_registry.make_env('wheel_legged',args=args,env_cfg=cfg)
    try:
        runner,_=task_registry.make_alg_runner(env=env,name='wheel_legged',args=args,train_cfg=train,log_root=None)
        runner.load(str(a.checkpoint),resume_mode='full')
        alg=runner.alg;model=alg.actor_critic
        alg.learning_rate=alg.optimizer.param_groups[0]['lr']
        obs,hist=env.get_observations();priv=env.get_privileged_observations()
        with torch.no_grad():
            for _ in range(1000):
                obs,priv,_,_,_,hist=env.step(model.act(obs,hist))
            for _ in range(runner.num_steps_per_env):
                action=alg.act(obs,hist,priv)
                obs,priv,reward,done,info,hist=env.step(action)
                alg.process_env_step(reward,done,info,obs)
            alg.compute_returns(torch.cat((priv,model.encode(hist)),dim=-1))
        store=alg.storage
        observations=store.observations.flatten(0,1)
        histories=store.observation_history.flatten(0,1)
        command=observations[:,8]/env.commands_scale[2]
        groups={}
        for h in [.375,.4,.425]:
            mask=torch.isclose(command,command.new_tensor(h))
            groups[str(h)]={'samples':int(mask.sum()),
                'mean_return':store.returns.flatten()[mask].mean().item(),
                'mean_value':store.values.flatten()[mask].mean().item(),
                'mean_advantage':store.advantages.flatten()[mask].mean().item(),
                'positive_advantage_fraction':(store.advantages.flatten()[mask]>0).float().mean().item()}
        o=observations[::24].clone();history=histories[::24].clone()
        def probe():
            result={}
            with torch.no_grad():
                for h in [.375,.4,.425]:
                    x=o.clone();y=history.clone().view(-1,5,25)
                    x[:,8]=h*env.commands_scale[2];y[:,:,8]=h*env.commands_scale[2]
                    action,_=model.act_inference(x,y.flatten(1))
                    result[h]=action.clone()
            return result
        before=probe();losses=alg.update();after=probe()
        payload={'checkpoint':str(a.checkpoint),'diagnostic_only':True,'grouped_rollout':groups,
                 'losses':list(losses),'learning_rate_after':alg.learning_rate,
                 'before_height_action_delta':(before[.425]-before[.375]).mean(0).cpu().tolist(),
                 'after_height_action_delta':(after[.425]-after[.375]).mean(0).cpu().tolist(),
                 'mean_action_change':{str(h):(after[h]-before[h]).mean(0).cpu().tolist() for h in before},
                 'limitations':'One sampled rollout/update; grouped advantages do not isolate causal credit. Counterfactual probes only, no policy checkpoint saved.'}
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2)+'\n')
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
