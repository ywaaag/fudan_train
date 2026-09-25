# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Frozen original body property randomization for RNG/state equivalence."""
import numpy as np
import torch

class OriginalBodyProperties:
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                self.base_add_mass = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_add_mass - min_add_mass)
                    + min_add_mass
                )
                self.base_mass = props[0].mass + self.base_add_mass
            props[0].mass += self.base_add_mass[env_id]
        else:
            self.base_mass[:] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                self.base_com[:, 0] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_x * 2)
                    - com_x
                )
                self.base_com[:, 1] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_y * 2)
                    - com_y
                )
                self.base_com[:, 2] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_z * 2)
                    - com_z
                )
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        return props

