"""Environment origin layout independent of simulator handles."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass(frozen=True)
class OriginLayout:
    custom_origins: bool
    env_origins: Optional[torch.Tensor] = None
    terrain_levels: Optional[torch.Tensor] = None
    terrain_types: Optional[torch.Tensor] = None
    flat_idx: Optional[torch.Tensor] = None
    smooth_slope_idx: Optional[torch.Tensor] = None
    rough_slope_idx: Optional[torch.Tensor] = None
    stair_up_idx: Optional[torch.Tensor] = None
    stair_down_idx: Optional[torch.Tensor] = None
    discrete_idx: Optional[torch.Tensor] = None
    basic_terrain_idx: Optional[torch.Tensor] = None
    advanced_terrain_idx: Optional[torch.Tensor] = None
    max_terrain_level: Optional[int] = None
    terrain_origins: Optional[torch.Tensor] = None
    terrain_x_max: Optional[float] = None
    terrain_x_min: Optional[float] = None
    terrain_y_max: Optional[float] = None
    terrain_y_min: Optional[float] = None


def build_origin_layout(*, num_envs, device, terrain_config, spacing, terrain_origin_array=None):
    """Keep original terrain buckets, grid convention and random draw order."""
    terrain_levels = terrain_types = terrain_origins = None
    smooth_slope_idx = rough_slope_idx = None
    stair_up_idx = stair_down_idx = discrete_idx = None
    basic_terrain_idx = advanced_terrain_idx = None
    max_terrain_level = None
    terrain_x_max = terrain_x_min = terrain_y_max = terrain_y_min = None
    if terrain_config.mesh_type in ["heightfield", "trimesh"]:
        custom_origins = True
        env_origins = torch.zeros(
            num_envs, 3, device=device, requires_grad=False
        )
        max_init_level = terrain_config.max_init_terrain_level
        if not terrain_config.curriculum:
            max_init_level = terrain_config.num_rows - 1
        terrain_levels = torch.randint(
            0, max_init_level + 1, (num_envs,), device=device
        )
        terrain_types = torch.div(
            torch.arange(num_envs, device=device),
            (num_envs / terrain_config.num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        flat_idx = (terrain_types < 4).nonzero(as_tuple=False).flatten()
        smooth_slope_idx = (
            ((4 <= terrain_types) * (terrain_types < 8))
            .nonzero(as_tuple=False)
            .flatten()
        )
        rough_slope_idx = (
            ((8 <= terrain_types) * (terrain_types < 12))
            .nonzero(as_tuple=False)
            .flatten()
        )
        stair_up_idx = (
            ((12 <= terrain_types) * (terrain_types < 14))
            .nonzero(as_tuple=False)
            .flatten()
        )
        stair_down_idx = (
            ((14 <= terrain_types) * (terrain_types < 18))
            .nonzero(as_tuple=False)
            .flatten()
        )
        discrete_idx = (
            ((18 <= terrain_types) * (terrain_types < 20))
            .nonzero(as_tuple=False)
            .flatten()
        )
        basic_terrain_idx = torch.cat(
            (
                flat_idx,
                smooth_slope_idx,
                rough_slope_idx,
                stair_down_idx,
            )
        )
        advanced_terrain_idx = torch.cat(
            (stair_up_idx, discrete_idx)
        )
        max_terrain_level = terrain_config.num_rows
        terrain_origins = (
            torch.from_numpy(terrain_origin_array)
            .to(device)
            .to(torch.float)
        )
        env_origins[:] = terrain_origins[
            terrain_levels, terrain_types
        ]
        terrain_x_max = (
            terrain_config.num_rows * terrain_config.terrain_length
            + terrain_config.border_size
        )
        terrain_x_min = -terrain_config.border_size
        terrain_y_max = (
            terrain_config.num_cols * terrain_config.terrain_length
            + terrain_config.border_size
        )
        terrain_y_min = -terrain_config.border_size
    else:
        custom_origins = False
        env_origins = torch.zeros(
            num_envs, 3, device=device, requires_grad=False
        )
        num_cols = np.floor(np.sqrt(num_envs))
        num_rows = np.ceil(num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        env_origins[:, 0] = spacing * xx.flatten()[: num_envs]
        env_origins[:, 1] = spacing * yy.flatten()[: num_envs]
        env_origins[:, 2] = 0.0
        flat_idx = torch.arange(num_envs, device=device)
    return OriginLayout(
        custom_origins=custom_origins,
        env_origins=env_origins,
        terrain_levels=terrain_levels,
        terrain_types=terrain_types,
        flat_idx=flat_idx,
        smooth_slope_idx=smooth_slope_idx,
        rough_slope_idx=rough_slope_idx,
        stair_up_idx=stair_up_idx,
        stair_down_idx=stair_down_idx,
        discrete_idx=discrete_idx,
        basic_terrain_idx=basic_terrain_idx,
        advanced_terrain_idx=advanced_terrain_idx,
        max_terrain_level=max_terrain_level,
        terrain_origins=terrain_origins,
        terrain_x_max=terrain_x_max,
        terrain_x_min=terrain_x_min,
        terrain_y_max=terrain_y_max,
        terrain_y_min=terrain_y_min,
    )
