import argparse
import os
import random
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ddpg.agent import DDPGAgent  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    set_seed(args.seed)
    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        initial_random_steps=args.initial_random_steps,
    )

    episode_rewards = []
    state, _ = env.reset(seed=args.seed)
    agent.reset_noise()
    episode_reward = 0.0
    episode = 0

    for step in range(1, args.total_steps + 1):
        action = agent.select_action(state, step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward

        if agent.ready_to_update() and step > args.initial_random_steps:
            agent.update()

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            if episode % args.log_every == 0:
                avg = np.mean(episode_rewards[-args.log_every:])
                print(f"episode {episode:4d} | step {step:6d} | avg reward (last {args.log_every}): {avg:8.1f}")
            state, _ = env.reset()
            agent.reset_noise()
            episode_reward = 0.0

    env.close()

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    agent.save(args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("DDPG on Pendulum-v1")
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plt.savefig(args.plot)
        print(f"saved reward curve to {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a DDPG agent on Pendulum-v1")
    p.add_argument("--total-steps", type=int, default=40_000)
    p.add_argument("--initial-random-steps", type=int, default=2_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint", type=str, default="checkpoints/ddpg_pendulum.pt")
    p.add_argument("--plot", type=str, default="assets/training_curve.png")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
