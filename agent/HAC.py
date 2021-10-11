import numpy as np
import torch

from agent.DDPG import DDPG
from agent.utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    def __init__(self, env, config, render=False):
        # init
        self.render = render
        self.env = env
        self.set_parameters(config["Parameter"])

        state_dim = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
        action_dim = env.action_space.shape[0]
        action_bounds = 0.5 * (self.action_clip_high - self.action_clip_low)
        action_offset = 0.5 * (self.action_clip_high + self.action_clip_low)
        state_bounds = 0.5 * (self.state_clip_high - self.state_clip_low)
        state_offset = 0.5 * (self.state_clip_high + self.state_clip_high)

        # adding lowest level
        self.HAC = [DDPG(state_dim, action_dim, action_bounds, action_offset, self.lr, self.H)]
        self.replay_buffer = [ReplayBuffer()]

        # adding remaining levels
        for _ in range(self.k_level - 1):
            self.HAC.append(DDPG(state_dim, state_dim, state_bounds, state_offset, self.lr, self.H))
            self.replay_buffer.append(ReplayBuffer())

        # set some parameters
        self.action_dim = action_dim
        self.state_dim = state_dim

        # logging parameters
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0

    def set_parameters(self, config):
        # k_level, H, state_dim, action_dim, render, threshold, action_bounds, action_offset, state_bounds, state_offset, lr
        # lamda, gamma, action_clip_low, action_clip_high, state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise

        self.k_level = int(config["k_level"])
        self.H = int(config["H"])
        self.lamda = float(config["lamda"])

        self.lr = float(config["lr"])
        self.gamma = float(config["gamma"])
        self.n_iter = int(config["n_iter"])
        self.batch_size = int(config["batch_size"])

        self.action_clip_low = self.env.action_space.low
        self.action_clip_high = self.env.action_space.high
        self.state_clip_low = self.env.observation_space["observation"].low
        self.state_clip_high = self.env.observation_space["observation"].high


    def run_HAC(self, i_level, state, goal, is_subgoal_test):
        next_state = None
        done = None
        goal_transitions = []

        # logging updates
        self.goals[i_level] = goal

        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test

            action = self.HAC[i_level].select_action(state, goal)

            #   <================ high level policy ================>
            if i_level > 0:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_state_noise)
                        action = action.clip(self.state_clip_low, self.state_clip_high)
                    else:
                        action = np.random.uniform(self.state_clip_low, self.state_clip_high)

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True

                # Pass subgoal to lower level
                next_state, done = self.run_HAC(i_level - 1, state, action, is_next_subgoal_test)

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.check_goal(action, next_state, self.threshold):
                    self.replay_buffer[i_level].add((state, action, -self.H, next_state, goal, 0.0, float(done)))

                # for hindsight action transition
                action = next_state

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

                    # if self.k_level == 2:
                    #     self.env.unwrapped.render_goal(self.goals[0], self.goals[1])
                    # elif self.k_level == 3:
                    #     self.env.unwrapped.render_goal_2(self.goals[0], self.goals[1], self.goals[2])

                    for _ in range(1000000):
                        continue

                # this is for logging
                self.reward += rew
                self.timestep += 1

            #   <================ finish one step/transition ================>

            # check if goal is achieved
            goal_achieved = self.check_goal(next_state, goal, self.threshold)

            # hindsight action transition
            if goal_achieved:
                self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
            else:
                self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))

            # copy for goal transition
            goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])

            state = next_state

            if done or goal_achieved:
                break

        #   <================ finish H attempts ================>

        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state
            self.replay_buffer[i_level].add(tuple(transition))

        return next_state, done

    def train(self, max_episodes, save_episode):
        # save trained models
        directory = "./preTrained/{}/{}level/".format("PandaStack", self.k_level)
        filename = "HAC_{}".format("PandaStack")

        # training procedure
        for i_episode in range(1, max_episodes + 1):
            self.reward = 0
            self.timestep = 0

            state = self.env.reset()
            print(state)
            # collecting experience in environment
            last_state, done = self.run_HAC(self.k_level - 1, state['observation'], state['desired_goal'], False)

            # update all levels
            self.update(self.n_iter, self.batch_size)

            if i_episode % save_episode == 0:
                self.save(directory, filename)

            print("Episode: {}\t Reward: {}".format(i_episode, self.reward))

    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)

    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + "_level_{}".format(i))

    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + "_level_{}".format(i))
