"""Compare a PyTorch sequence actor with its exported ONNX actor."""

import argparse
import os

import numpy as np
import torch

from wheel_legged_gym.learning.modules.actor_critic_sequence import ActorCriticSequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    import onnxruntime as ort

    model = ActorCriticSequence(
        num_obs=25,
        num_critic_obs=1,
        num_actions=6,
        num_encoder_obs=125,
        latent_dim=3,
        encoder_hidden_dims=[128, 64],
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = {k: v for k, v in checkpoint["model_state_dict"].items() if not k.startswith("critic.")}
    model.load_state_dict(state, strict=False)

    rng = np.random.default_rng(1234)
    obs = rng.standard_normal((args.batch, 25), dtype=np.float32)
    history = rng.standard_normal((args.batch, 125), dtype=np.float32)
    with torch.no_grad():
        torch_actions = model.actor(torch.cat([obs_tensor := torch.from_numpy(obs), model.encoder(torch.from_numpy(history))], dim=-1))

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_actions = session.run(["actions"], {"obs": obs, "obs_history": history})[0]
    error = np.abs(torch_actions.numpy() - onnx_actions)
    print(f"checkpoint={os.path.abspath(args.checkpoint)}")
    print(f"onnx={os.path.abspath(args.onnx)}")
    print(f"max_abs_error={error.max():.9g}")
    print(f"mean_abs_error={error.mean():.9g}")
    if not np.allclose(torch_actions.numpy(), onnx_actions, rtol=1e-4, atol=1e-5):
        raise SystemExit("FAIL: PyTorch and ONNX actions differ")
    print("PASS: PyTorch and ONNX actions match")


if __name__ == "__main__":
    main()
