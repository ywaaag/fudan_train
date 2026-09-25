"""Training command schedule, with no action or state compensation."""
import torch


def alternate_height(commands, episode_steps, dt, interval_s, low, high):
    """Flip only non-.40 height slots; preserve vx/yaw and all retention slots."""
    interval=round(interval_s/dt)
    if interval<1 or not low<.4<high:raise ValueError('Invalid alternating-height schedule')
    mask=(episode_steps>0)&(episode_steps%interval==0)&(~torch.isclose(commands[:,2],commands.new_tensor(.4)))
    commands[mask,2]=torch.where(commands[mask,2]<.4,commands.new_tensor(high),commands.new_tensor(low))
