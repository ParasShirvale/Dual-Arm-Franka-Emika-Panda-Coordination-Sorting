import os
import sys
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# add repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

from ppo_env import SingleArmRL, DualArmRL, Handover

def train():
    # ---------- Environment ----------
    env = make_vec_env(
        DualArmRL,
        n_envs=100  # MuJoCo + viewer → keep 1
    )

    # ---------- PPO Model ----------
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=4096, 
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="auto"
    )

    # ---------- Train ----------
    model.learn(total_timesteps=200_000)

    # ---------- Save ----------
    model.save("file_name")

    env.close()


if __name__ == "__main__":
    train()
