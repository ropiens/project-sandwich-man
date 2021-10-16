from ast import literal_eval

import numpy as np
import torch

from agent.DDPG import DDPG
from agent.utils import ReplayBuffer, distance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    def __init__(self, env, config, render=False):
        # init
        self.render = render
        self.env = env

        self.set_parameters(config["Parameter"])

        action_bounds = 0.5 * (self.action_clip_high - self.action_clip_low)
        action_offset = 0.5 * (self.action_clip_high + self.action_clip_low)
        goal_bounds = 0.5 * (self.goal_clip_high - self.goal_clip_low)
        goal_offset = 0.5 * (self.goal_clip_high + self.goal_clip_low)

        # adding lowest level
        self.HAC = [DDPG(self.state_dim, self.goal_dim, self.action_dim, action_bounds, action_offset, self.lr, self.H)]
        self.replay_buffer = [ReplayBuffer()]

        # adding remaining levels
        for _ in range(self.k_level - 1):
            self.HAC.append(DDPG(self.state_dim, self.goal_dim, self.goal_dim, goal_bounds, goal_offset, self.lr, self.H))
            self.replay_buffer.append(ReplayBuffer())

        # logging parameters
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0

        # for subgoal viz
        self.subgoal_1, self.subgoal_2 = None, None

    def set_parameters(self, config):
        # environment dependent parameters
        self.state_dim = self.env.observation_space["observation"].shape[0]
        self.goal_dim = self.env.observation_space["desired_goal"].shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.action_clip_low = self.env.action_space.low
        self.action_clip_high = self.env.action_space.high
        self.goal_clip_low = np.concatenate(
            (np.array(literal_eval(config["workspace_low"])), np.array(literal_eval(config["workspace_low"])))
        )
        self.goal_clip_high = np.concatenate(
            (np.array(literal_eval(config["workspace_high"])), np.array(literal_eval(config["workspace_high"])))
        )

        # HAC parameters
        self.k_level = int(config["k_level"])
        self.H = int(config["H"])
        self.lamda = float(config["lamda"])
        # DDPG parameters
        self.lr = float(config["lr"])
        self.gamma = float(config["gamma"])
        self.n_iter = int(config["n_iter"])
        self.batch_size = int(config["batch_size"])

        self.threshold = float(config["threshold"])

        # exploration noise
        self.exploration_action_noise = np.array([float(config["exploration_action_noise"])] * self.action_dim)
        self.exploration_goal_noise = np.array([float(config["exploration_goal_noise"])] * self.goal_dim)

    def render_subgoal(self, subgoal):
        p = self.env.sim.physics_client

        if not self.subgoal_1 is None:
            p.removeBody(self.subgoal_1)
        if not self.subgoal_2 is None:
            p.removeBody(self.subgoal_2)

        # visualize subgoal_1
        sphere_1 = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.9, 0.1, 0.1, 0.3],
            specularColor=[0, 0, 0],
            visualFramePosition=subgoal[:3],
        )

        self.subgoal_1 = p.createMultiBody(baseVisualShapeIndex=sphere_1)

        # visualize subgoal_2
        sphere_2 = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.1, 0.9, 0.1, 0.3],
            specularColor=[0, 0, 0],
            visualFramePosition=subgoal[3:],
        )

        self.subgoal_2 = p.createMultiBody(baseVisualShapeIndex=sphere_2)

    def check_goal(self, achieved_goal, desired_goal):
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return (d < self.threshold).astype(np.float32)

    def run_HAC(self, i_level, state, goal, is_subgoal_test, debug=False):
        next_state = None
        done = None
        goal_transitions = []

        # logging updates
        self.goals[i_level] = goal

        # H attempts
        for iter in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test

            if debug:
                print(f"\033[{i_level+40}m" + f"lev:{i_level}\niter:{iter}\n state:{state}\n goal:{goal}" + "\033[0m")
            action = self.HAC[i_level].select_action(state, goal)

            #   <================ high level policy ================>
            if i_level > 0:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_goal_noise)
                        action = action.clip(self.goal_clip_low, self.goal_clip_high)
                    else:
                        action = np.random.uniform(self.goal_clip_low, self.goal_clip_high)

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True

                # Pass subgoal to lower level
                next_state, done = self.run_HAC(i_level - 1, state, action, is_next_subgoal_test)

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.check_goal(next_state["achieved_goal"], action):
                    self.replay_buffer[i_level].add((state, action, -self.H, next_state["observation"], goal, 0.0, float(done)))

                # for hindsight action transition
                action = next_state["achieved_goal"]

            #   <================ low level policy ================>
            else:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_action_noise)
                        action = action.clip(self.action_clip_low, self.action_clip_high)
                    else:
                        action = np.random.uniform(self.action_clip_low, self.action_clip_high)

                # take primitive action
                next_state, rew, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()
                    self.render_subgoal(goal)

                    for _ in range(1000000):
                        continue

                # this is for logging
                self.reward += rew
                self.timestep += 1

            #   <================ finish one step/transition ================>
            # check if goal is achieved
            goal_achieved = self.check_goal(next_state["achieved_goal"], goal)

            # hindsight action transition
            if goal_achieved:
                self.replay_buffer[i_level].add((state, action, 0.0, next_state["observation"], goal, 0.0, float(done)))
            else:
                self.replay_buffer[i_level].add((state, action, -1.0, next_state["observation"], goal, self.gamma, float(done)))

            # copy for goal transition
            goal_transitions.append([state, action, -1.0, next_state["observation"], None, self.gamma, float(done)])

            state = next_state["observation"]

            if done or goal_achieved:
                break

        #   <================ finish H attempts ================>

        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state["achieved_goal"]
            self.replay_buffer[i_level].add(tuple(transition))

        return next_state, done

    def train(self, max_episodes, save_episode, save_path=("", "")):

        # training procedure
        for i_episode in range(1, max_episodes + 1):
            self.reward = 0
            self.timestep = 0

            state = self.env.reset()
            # collecting experience in environment
            last_state, done = self.run_HAC(self.k_level - 1, state["observation"], state["desired_goal"], False)

            # update all levels
            self.update(self.n_iter, self.batch_size)

            if i_episode % save_episode == 0:
                self.save(save_path)

            print("Episode: {}\t Reward: {}".format(i_episode, self.reward))

    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)

    def save(self, save_path):
        directory, name = save_path
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + "_level_{}".format(i))

    def load(self, save_path):
        directory, name = save_path
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + "_level_{}".format(i))
