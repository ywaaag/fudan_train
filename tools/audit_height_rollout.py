"""Observe actual training command coverage and reward contributions without updates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plane'))
import isaacgym
import torch
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.scripts.isaac_parity_trace import _gym_args
from wheel_legged_gym.scripts.isaac_command_grid import load_policy


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cfg,train=task_registry.get_cfgs('wheel_legged')
    apply_policy_experiment(cfg,'HEIGHT_COURSE',train)
    cfg.env.num_envs=len(cfg.commands.height_bank)*4
    args=_gym_args();args.num_envs=cfg.env.num_envs;args.seed=23
    env,_=task_registry.make_env('wheel_legged',args=args,env_cfg=cfg)
    try:
        model=load_policy(a.checkpoint,env.device)
        env.reset()
        obs,hist=env.get_observations()
        heights=sorted({row[2] for row in cfg.commands.height_bank})
        names=['samples','height','height_mae','vx_mae','resets']+list(env.reward_scales)
        sums={h:dict.fromkeys(names,0.) for h in heights}
        coverage={}
        sensitivity=[]
        switches=0
        for step in range(1500):
            assert torch.allclose(obs[:,6:9],env.commands[:,:3]*env.commands_scale,atol=1e-6)
            command=env.commands.clone()
            if step in (500,1000,1499):
                # Same real physical observation/history, change only the height command.
                # Counterfactual diagnostic, never sent to the simulator as control.
                actions=[]
                with torch.no_grad():
                    for h in heights:
                        probe_obs=obs.clone();probe_hist=hist.clone().view(env.num_envs,5,25)
                        probe_obs[:,8]=h*env.commands_scale[2]
                        probe_hist[:,:,8]=h*env.commands_scale[2]
                        mean,_=model.act_inference(probe_obs,probe_hist.reshape(env.num_envs,125))
                        actions.append(mean)
                delta=actions[-1]-actions[0]
                sensitivity.append({'step':step,'height_span_m':heights[-1]-heights[0],
                    'mean_action_delta':delta.mean(dim=0).cpu().tolist(),
                    'mean_abs_action_delta':delta.abs().mean(dim=0).cpu().tolist(),
                    'action_std':model.std.detach().cpu().tolist()})
            with torch.no_grad(): action=model.act(obs,hist)
            obs,_,_,done,_,hist=env.step(action)
            changed=~torch.isclose(command[:,2],env.commands[:,2])
            continuing=changed & ~done.bool()
            if continuing.any():
                assert torch.allclose(command[continuing,:2],env.commands[continuing,:2])
                assert not torch.isclose(command[continuing,2],command.new_tensor(.4)).any()
                switches+=int(continuing.sum())
            if step<500:continue
            for h in heights:
                mask=torch.isclose(command[:,2],command.new_tensor(h))
                n=mask.sum().item();d=sums[h];d['samples']+=n
                d['height']+=env.base_height[mask].sum().item()
                d['height_mae']+=(env.base_height[mask]-h).abs().sum().item()
                d['vx_mae']+=(env.base_lin_vel[mask,0]-command[mask,0]).abs().sum().item()
                d['resets']+=done[mask].sum().item()
                for key,scale in env.reward_scales.items():
                    value=getattr(env,'_reward_'+key)()*scale
                    if key not in getattr(cfg.rewards,'unclipped_reward_names',()):
                        bound=cfg.rewards.clip_single_reward*env.dt
                        value=value.clamp(-bound,bound)
                    d[key]+=value[mask].sum().item()/env.dt
            if step%100==0:
                unique,counts=torch.unique(command[:,:3],dim=0,return_counts=True)
                for row,count in zip(unique.cpu().tolist(),counts.cpu().tolist()):
                    key=','.join(f'{v:.3f}' for v in row);coverage[key]=coverage.get(key,0)+count
        for d in sums.values():
            for key in names:
                if key not in ['samples','resets']:d[key]/=max(1,d['samples'])
        payload={'checkpoint':str(a.checkpoint),'diagnostic_only':True,'action':'sampled','noise':cfg.noise.add_noise,
                 'coverage_snapshots':coverage,'by_height':sums,'height_action_sensitivity':sensitivity,
                 'height_switches_without_reset':switches,
                 'note':'Training profile rollout, no updates. Reward functions recomputed post-step; reset samples may differ from training reward timing.'}
        if getattr(cfg.commands,'height_switch_interval',0)>0 and switches==0:
            raise RuntimeError('Requested height schedule produced no observed switches')
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2)+'\n')
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
