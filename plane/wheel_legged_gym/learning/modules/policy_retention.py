"""Training-only reference loss on retention environments, never runtime control."""
import torch


def reference_loss(mean,reference_mean,reference_std,env_ids,dynamic_stride):
    if dynamic_stride not in (2,4):raise ValueError('Expected reviewed dynamic cohort stride')
    retained=env_ids.remainder(dynamic_stride)!=0
    if not retained.any():return mean.sum()*0.,retained.float().mean()
    delta=(mean[retained]-reference_mean.detach()[retained])/reference_std.detach().clamp_min(1e-4)
    return .5*delta.square().mean(),retained.float().mean()
