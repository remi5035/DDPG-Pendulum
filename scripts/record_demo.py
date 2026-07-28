import argparse
import os
import sys

import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ddpg.agent import DDPGAgent  # noqa: E402


def record(args):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    agent.load(args.checkpoint)

    os.makedirs(args.video_folder, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.video_folder,
        episode_trigger=lambda ep: True,
        name_prefix=args.name_prefix,
    )

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        done = False
        step = 0
        while not done:
            action = agent.select_action(state, step=0, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        print(f"episode {episode}: reward = {episode_reward:.1f}")

    env.close()


def parse_args():
    p = argparse.ArgumentParser(description="Record a video of a trained DDPG agent on Pendulum-v1")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ddpg_pendulum.pt")
    p.add_argument("--video-folder", type=str, default="assets/videos")
    p.add_argument("--name-prefix", type=str, default="ddpg-pendulum")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    record(parse_args())
