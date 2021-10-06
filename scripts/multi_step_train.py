import argparse
import configparser
import os

import gym
import numpy as np
import panda_gym
import torch

from agent import HAC as trainer


def main(args: argparse.ArgumentParser, config: configparser.ConfigParser):
    """Main Function to launch multi-step trainer"""
    env = gym.make(args.env_id, render=False if args.no_render else True)

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

    # Initialize HAC agent and setting parameters
    agent = trainer.HAC(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-step RL Trainer Arguments using HAC")
    config = configparser.ConfigParser()

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
        help="gym render option",
    )
    args = parser.parse_args()
    config.read("./config/multi_step_agent.cfg")

    main(args, config)
