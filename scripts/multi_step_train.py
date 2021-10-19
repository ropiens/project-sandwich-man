import argparse
import configparser
import os
from ast import literal_eval

import gym
import numpy as np
import panda_gym
import torch
from gym.core import Env

from agent import HAC as trainer


def visualize_workspace(env: Env, config: configparser.ConfigParser) -> None:
    """
    Visualize workspace(sub-goal range)

    Args:
        env(Env): gym environment(including robot & task)

        config(ConfigParser): parsed data(parameters) from configuration file
    """
    config = config["Parameter"]
    workspace_high = literal_eval(config["goal_clip_high"])

    env.sim.create_box(
        body_name="workspace",
        half_extents=[workspace_high[0] * 2, workspace_high[1] * 2, workspace_high[1]],
        mass=0,
        position=[0.0, 0.0, workspace_high[1]],
        specular_color=[0.0, 0.0, 0.0],
        rgba_color=[0.0, 0.5, 0.5, 0.15],
        ghost=True,
    )


def add_landmark(env: Env) -> None:
    """
    Add Ropiens Landmark on gym env

    Args:
        env(Env): gym environment(including robot & task)
    """
    p = env.sim.physics_client
    path = os.path.abspath(os.path.join(__file__, "../.."))
    p.loadURDF(
        f"{path}/ropiens/ropiens.urdf",
        [-0.2, 0.9, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        globalScaling=3,
        useFixedBase=1,
    )


def main(args: argparse.ArgumentParser, config: configparser.ConfigParser) -> None:
    """
    Main Function to launch multi-step trainer

    Args:
        args(ArgumentParser): some arguments from users(including env_id, render etc.)

        config(ConfigParser): parsed data(parameters) from configuration file
    """
    env = gym.make(args.env_id, render=False if args.no_render else True)

    # Add our Landmark
    add_landmark(env)

    # Visualize workspace
    visualize_workspace(env, config)

    # Initialize HAC agent and setting parameters
    agent = trainer.HAC(env, config, render=False if args.no_render else True)

    # save_path of trained models
    path = os.path.abspath(os.path.join(__file__, "../.."))
    directory = f"{path}/pretrained/{args.env_id}/"
    filename = f"HAC_{args.env_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = (directory, filename)

    # train
    agent.train(max_episodes=1000, save_episode=10, save_path=save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-step RL Trainer Arguments using HAC")
    config = configparser.ConfigParser()

    parser.add_argument(
        "--env-id",
        type=str,
        default="PandaStack-v1",
        help="Environment default: PandaStack-v1, \
        option :[PandaReach-v1(not allowed), PandaSlice-v1(not allowed), \
        PandaPush-v1(not allowed), PandaPickAndPlace-v1(not allowed), PandaStack-v1]",
    )
    parser.add_argument(
        "--no-render",
        type=bool,
        default=False,
        help="gym render option",
    )
    args = parser.parse_args()

    if not args.env_id == "PandaStack-v1":
        print(f"\033[{41}m" + "\nNot allowed now. you can choose 'PandaStack-v1' only.\n" + "\033[0m")
        args.env_id = "PandaStack-v1"

    path = os.path.abspath(os.path.join(__file__, "../.."))
    config.read(f"{path}/agent/config/multi_step_agent.cfg")

    main(args, config)
