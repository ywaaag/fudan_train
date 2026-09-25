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

"""Frozen shape randomization for exact value/RNG comparisons."""
from isaacgym.torch_utils import torch_rand_float
import torch

class OriginalShapes:
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (num_buckets, 1),
                    device=self.device,
                )
                self.friction_coef = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coef[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                (
                    min_restitution,
                    max_restitution,
                ) = self.cfg.domain_rand.restitution_range
                self.restitution_coef = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_restitution - min_restitution)
                    + min_restitution
                )
            for s in range(len(props)):
                props[s].restitution = self.restitution_coef[env_id]
        return props

