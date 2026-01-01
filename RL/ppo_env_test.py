import os
import sys
import time
import mujoco
import mujoco.viewer
from ppo_env import SingleArmRL , DualArmRL, Handover # adjust import path
# from two_arms import DecisionRoboticEnv

# Setup paths to ensure imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)


def test_env(ep):
    env = SingleArmRL()
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"\n===== Episode {ep + 1} =====")
    print("Initial obs:", obs)

    while not done:
        # Random action (for testing only)
        action = env.action_space.sample()
        
        print(
            f"Step {step:02d} | "
            f"Action: {action} | "
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        print("    "
            f"Obs: {obs} | "
            f"Reward: {reward:.2f}"
        )

        # Slow down to observe (optional)
        time.sleep(0.05)

        print(f"Episode finished in {step} steps")
        print(f"Total reward: {total_reward:.2f}")

    viewer.close()
    env.close()

if __name__ == "__main__":
    
    for ep in range(5):
        test_env(ep)
