"""Export a trained PPO policy to ONNX for backend-free, in-browser inference.

    python -m raceline.export_onnx --checkpoint runs/ppo/best_model.zip --out web/policy.onnx

SB3's PPO actor is an MLP feature extractor + policy net producing action logits. For a
discrete action space the deployed policy is ``argmax(logits)`` of a deterministic forward
pass, so we export a tiny module mapping observation -> action logits and run argmax in JS.
The legacy TorchScript exporter is used so the graph uses only ops onnxruntime-web supports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO

from raceline.envs.racetrack_env import RaceTrackEnv


class _LogitsPolicy(torch.nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        self.policy = sb3_policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def export(checkpoint: str, out: str) -> Path:
    model = PPO.load(checkpoint, device="cpu")
    model.policy.eval()
    wrapper = _LogitsPolicy(model.policy)
    wrapper.eval()

    obs_dim = model.observation_space.shape[0]
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)

    # Parity check: argmax of exported logits must match SB3's deterministic predict.
    env = RaceTrackEnv()
    obs, _ = env.reset(seed=4242)
    obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        my_action = int(torch.argmax(wrapper(obs_t), dim=1).item())
    sb3_action, _ = model.predict(obs, deterministic=True)
    assert my_action == int(sb3_action), f"export mismatch: {my_action} vs {int(sb3_action)}"

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={"observation": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy exporter -> standard ops onnxruntime-web fully supports
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"exported -> {out_path} ({size_kb:.1f} KB, obs_dim={obs_dim})")
    print(f"parity check passed: exported argmax == SB3 predict (action {my_action})")
    _verify_onnxruntime(out_path, obs)
    return out_path


def _verify_onnxruntime(path: Path, obs: np.ndarray) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("(onnxruntime not installed; skipping runtime verification)")
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"observation": obs.reshape(1, -1).astype(np.float32)})[0]
    print(f"onnxruntime check: logits shape {logits.shape}, argmax {int(np.argmax(logits))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PPO policy to ONNX")
    ap.add_argument("--checkpoint", default="runs/ppo/best_model.zip")
    ap.add_argument("--out", default="web/policy.onnx")
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
