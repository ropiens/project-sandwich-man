import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import panda_gym
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = "best_model"
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-10:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(f"{self.save_path}/{self.save_path}")

        return True


def main(args):
    # for the reward log
    currpath = os.getcwd()

    log_dir = currpath
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args.env_id, render=False if args.no_render else True)
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=100, log_dir=log_dir
    )  # check frequency set for reward checking

    her_kwargs = dict(
        online_sampling=True, n_sampled_goal=4, goal_selection_strategy="future", max_episode_length=100
    )

    # Add our Landmark
    p = env.sim.physics_client
    path = os.path.abspath(os.path.join(__file__, "../.."))
    p.loadURDF(
        f"{path}/ropiens/ropiens.urdf",
        [-0.2, 0.9, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        globalScaling=3,
        useFixedBase=1,
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        ent_coef="auto",
        replay_buffer_kwargs=her_kwargs,
        verbose=1,
        buffer_size=1000000,
        learning_rate=0.001,
        learning_starts=1000,
        gamma=0.95,
        batch_size=256,
        tensorboard_log=f"./logs/sac_her/{args.env_id}",
        policy_kwargs=dict(net_arch=[128, 128, 128]),
    )

    try:
        model.learn(total_timesteps=2e2, callback=callback)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        model.save(str(currpath) + f"/logs/sac_her/{args.env_id}")
        model.save_replay_buffer(str(currpath) + f"/logs/sac_her/{args.env_id}/replay_buffer")

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Single-step RL Trainer using SAC+HER")

    parser.add_argument(
        "--env-id",
        type=str,
        default="PandaReach-v1",
        help="panda-gym environment default: PandaReach-v1, \
        option :[PandaReach-v1, PandaSlice-v1, PandaPush-v1, PandaPickAndPlace-v1, PandaStack-v1]",
    )
    parser.add_argument(
        "--no-render",
        type=bool,
        default=False,
        help="gym render otpion",
    )
    args = parser.parse_args()

    main(args)
