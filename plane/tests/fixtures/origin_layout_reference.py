"""Frozen environment origin builder before extraction, 2026-09-24."""
import numpy as np
import torch

def _get_env_origins(self):
    """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
    Otherwise create a grid.
    """
    if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        self.custom_origins = True
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level
        if not self.cfg.terrain.curriculum:
            max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(
            0, max_init_level + 1, (self.num_envs,), device=self.device
        )
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            (self.num_envs / self.cfg.terrain.num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        # num_cols = 20
        # terrain types: [flat, smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain types: [0 1 2 3, 4 5 6 7, 8 9 10 11, 12 13, 14 15 16 17, 18 19]
        # terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.2, 0.1]
        self.flat_idx = (self.terrain_types < 4).nonzero(as_tuple=False).flatten()
        self.smooth_slope_idx = (
            ((4 <= self.terrain_types) * (self.terrain_types < 8))
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.rough_slope_idx = (
            ((8 <= self.terrain_types) * (self.terrain_types < 12))
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.stair_up_idx = (
            ((12 <= self.terrain_types) * (self.terrain_types < 14))
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.stair_down_idx = (
            ((14 <= self.terrain_types) * (self.terrain_types < 18))
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.discrete_idx = (
            ((18 <= self.terrain_types) * (self.terrain_types < 20))
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.basic_terrain_idx = torch.cat(
            (
                self.flat_idx,
                self.smooth_slope_idx,
                self.rough_slope_idx,
                self.stair_down_idx,
            )
        )
        self.advanced_terrain_idx = torch.cat(
            (self.stair_up_idx, self.discrete_idx)
        )
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = (
            torch.from_numpy(self.terrain.env_origins)
            .to(self.device)
            .to(torch.float)
        )
        self.env_origins[:] = self.terrain_origins[
            self.terrain_levels, self.terrain_types
        ]
        self.terrain_x_max = (
            self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length
            + self.cfg.terrain.border_size
        )
        self.terrain_x_min = -self.cfg.terrain.border_size
        self.terrain_y_max = (
            self.cfg.terrain.num_cols * self.cfg.terrain.terrain_length
            + self.cfg.terrain.border_size
        )
        self.terrain_y_min = -self.cfg.terrain.border_size
    else:
        self.custom_origins = False
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0
        self.flat_idx = torch.arange(self.num_envs, device=self.device)
