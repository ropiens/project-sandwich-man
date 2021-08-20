import os

import gym
import panda_gym


def test():
    """test panda-gym environment"""
    env = gym.make("PandaStack-v1", render=True)

    p = env.sim.physics_client
    path = os.path.abspath(os.path.join(__file__, "../.."))
    p.loadURDF(
        f"{path}/ropiens/ropiens.urdf",
        [-0.2, 0.9, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        globalScaling=3,
        useFixedBase=1,
    )

    obs = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        env.step(action)

    env.close()


if __name__ == "__main__":
    test()
