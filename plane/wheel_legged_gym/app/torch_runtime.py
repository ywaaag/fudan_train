"""Explicit process-wide Torch settings required by the legacy Isaac runtime."""
import torch


def configure_isaac_torch_runtime():
    """Apply before creating environments; preserve the established JIT settings.

    These switches are process-wide, not per-environment state. Do not call from
    domain functions or at import time. Direct environment users must call this
    entrypoint themselves; TaskRegistry.make_env already does so.
    """
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
