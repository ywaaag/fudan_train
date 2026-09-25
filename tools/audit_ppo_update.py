"""Single real rollout/update audit. Diagnostic only; saves no new policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plane'))
import isaacgym
import torch
from wheel_legged_gym.app.bootstrap import create_task_registry
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment
from wheel_legged_gym.adapters.isaacgym.evaluation_setup import build_evaluation_args


def main():
    task_registry = create_task_registry()
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--profile',choices=['H3_SPEED1','EXPLORE_STOP_RETENTION'],default='H3_SPEED1')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--warmup',type=int,default=1000,help='No-update policy steps before collecting 48-step rollout')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cfg,train=task_registry.get_cfgs('wheel_legged')
    apply_policy_experiment(cfg,a.profile,train)
    args=build_evaluation_args();args.num_envs=a.num_envs;args.seed=23
    cfg.env.num_envs=a.num_envs
    env,_=task_registry.make_env('wheel_legged',args=args,env_cfg=cfg)
    try:
        runner,_=task_registry.make_alg_runner(env=env,name='wheel_legged',args=args,train_cfg=train,log_root=None)
        runner.load(str(a.checkpoint),resume_mode='full')
        alg=runner.alg;model=alg.actor_critic
        alg.encoder_action_anchor_coef=1.
        alg.encoder_ablation_diagnostics=True
        obs,history=env.get_observations();priv=env.get_privileged_observations()
        with torch.no_grad():
            for _ in range(a.warmup):
                action=model.act(obs,history)
                obs,priv,_,_,_,history=env.step(action)
            for _ in range(runner.num_steps_per_env):
                action=alg.act(obs,history,priv)
                obs,priv,reward,done,info,history=env.step(action)
                alg.process_env_step(reward,done,info,obs)
            alg.compute_returns(torch.cat((priv,model.encode(history)),dim=-1))
        store=alg.storage
        commands=store.observations[:,:,6].flatten()/env.commands_scale[0]
        true_vx=store.privileged_observations[:,:,0].flatten()/env.obs_scales.lin_vel
        observations=store.observations.flatten(0,1)
        histories=store.observation_history.flatten(0,1)
        values=store.values.flatten();returns=store.returns.flatten();advantages=store.advantages.flatten()
        reward=store.rewards.flatten()
        groups={}
        selected=[]
        for v in [-1.,-.5,-.1,0.,.1,.5,1.]:
            mask=torch.isclose(commands,commands.new_tensor(v),atol=1e-5)
            ids=mask.nonzero().flatten();n=len(ids)
            if not n:continue
            selected.append(ids[:256])
            groups[str(v)]={'samples':n,'vx':true_vx[mask].mean().item(),
                'vx_mae':(true_vx[mask]-v).abs().mean().item(),
                'value':values[mask].mean().item(),'return':returns[mask].mean().item(),
                'raw_advantage':(returns-values)[mask].mean().item(),
                'normalized_advantage':advantages[mask].mean().item(),
                'positive_advantage_fraction':(advantages[mask]>0).float().mean().item(),
                'bootstrapped_reward_per_second':reward[mask].mean().item()/env.dt}
        ids=torch.cat(selected);probe_obs=observations[ids].clone();probe_hist=histories[ids].clone()
        probe_cmd=commands[ids].clone()
        def means():
            with torch.no_grad():return model.actor(torch.cat((probe_obs,model.encode(probe_hist)),dim=-1)).clone()
        initial=means();trace=[];clip_trace=[]
        original_step=alg.optimizer.step
        def step(*args,**kwargs):
            before=means()
            params=list(model.actor.parameters())
            old_params=[p.detach().clone() for p in params]
            gradients=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            result=original_step(*args,**kwargs)
            after=means()
            dot=sum((g*(p.detach()-old)).sum() for p,old,g in zip(params,old_params,gradients)).item()
            trace.append({'actor_gradient_dot_update':dot,
                'direction_note':'negative means first-order descent on current total loss; positive means momentum opposes current gradient',
                'by_command':{str(v):{'delta_mean_action':(after-before)[torch.isclose(probe_cmd,probe_cmd.new_tensor(v),atol=1e-5)].mean(dim=0).cpu().tolist()}
                for v in [-1.,-.5,-.1,0.,.1,.5,1.] if torch.isclose(probe_cmd,probe_cmd.new_tensor(v),atol=1e-5).any()}})
            return result
        alg.optimizer.step=step
        original_clip=torch.nn.utils.clip_grad_norm_
        def clip(parameters,max_norm,*args,**kwargs):
            params=list(parameters)
            record={'limit':float(max_norm)}
            for name,ps in [('actor',model.actor.parameters()),('critic',model.critic.parameters()),
                            ('encoder',model.encoder.parameters()),('std',[model.std])]:
                record[name]=sum(float(p.grad.detach().square().sum()) for p in ps if p.grad is not None)**.5
            value=original_clip(params,max_norm,*args,**kwargs)
            record['total_norm']=float(value);clip_trace.append(record)
            return value
        torch.nn.utils.clip_grad_norm_=clip
        try:losses=alg.update()
        finally:torch.nn.utils.clip_grad_norm_=original_clip
        final=means()
        for v,g in groups.items():
            mask=torch.isclose(probe_cmd,probe_cmd.new_tensor(float(v)),atol=1e-5)
            g['total_action_change']=(final-initial)[mask].mean(dim=0).cpu().tolist()
        out={'checkpoint':str(a.checkpoint.resolve()),'profile':a.profile,'num_envs':a.num_envs,
            'warmup_steps':a.warmup,'seed':23,'diagnostic_only':True,'policy_saved':False,
            'groups':groups,'gradient_clips':clip_trace,'actor_steps':trace,'losses':list(losses),
            'encoder_action_shift':alg.last_encoder_action_shift,
            'note':'GAE targets are bootstrapped predictions, not independent Monte Carlo truth; one short update cannot establish long-run causality.'}
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({'out':str(a.out),'groups':groups,'first_clips':clip_trace[:2]}))
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
