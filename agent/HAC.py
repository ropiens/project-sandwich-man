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
        self.writer = None

        self.set_parameters(config["Parameter"])

        action_bounds = 0.5 * (self.action_clip_high - self.action_clip_low)
        action_offset = 0.5 * (self.action_clip_high + self.action_clip_low)
        goal_bounds = 0.5 * (self.goal_clip_high - self.goal_clip_low)
        goal_offset = 0.5 * (self.goal_clip_high + self.goal_clip_low)

        # adding lowest level
        self.HAC = [
            DDPG(
                self.state_dim,
                self.goal_dim,
                self.action_dim,
                action_bounds,
                action_offset,
                self.lr,
                self.H,
                self.tau,
                name="level_0",
            )
        ]
        self.replay_buffer = [ReplayBuffer()]

        # adding remaining levels
        for i in range(1, self.k_level):
            if i == self.k_level - 1:
                final_goal_dim = self.env.observation_space["desired_goal"].shape[0]
                self.HAC.append(
                    DDPG(
                        self.state_dim,
                        final_goal_dim,
                        self.goal_dim,
                        goal_bounds,
                        goal_offset,
                        self.lr,
                        self.H,
                        self.tau,
                        name=f"level_{i}",
                    )
                )
                self.replay_buffer.append(ReplayBuffer())
            else:
                self.HAC.append(
                    DDPG(
                        self.state_dim,
                        self.goal_dim,
                        self.goal_dim,
                        goal_bounds,
                        goal_offset,
                        self.lr,
                        self.H,
                        name=f"level_{i}",
                    )
                )
                self.replay_buffer.append(ReplayBuffer())

        # logging parameters
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0

        # for subgoal viz
        self.create_subgoal()

    def set_parameters(self, config):
        # environment dependent parameters
        self.state_dim = self.env.observation_space["observation"].shape[0]
        self.goal_dim = (
            len(self.env.robot.get_ee_position()) + self.env.observation_space["desired_goal"].shape[0]
        )  # block-gripper-informed goal
        if not self.env.robot.block_gripper:
            self.goal_dim += 1
        self.action_dim = self.env.action_space.shape[0]

        self.action_clip_low = self.env.action_space.low
        self.action_clip_high = self.env.action_space.high
        self.goal_clip_low = np.array(literal_eval(config["goal_clip_low"]))
        self.goal_clip_high = np.array(literal_eval(config["goal_clip_high"]))

        # HAC parameters
        self.k_level = int(config["k_level"])
        self.H = int(config["H"])
        self.lamda = float(config["lamda"])
        # DDPG parameters
        self.lr = float(config["lr"])
        self.tau = float(config["tau"])
        self.gamma = float(config["gamma"])
        self.n_iter = int(config["n_iter"])
        self.batch_size = int(config["batch_size"])

        self.threshold = float(config["threshold"])

        # exploration noise
        self.exploration_action_noise = np.array([float(config["exploration_action_noise"])] * self.action_dim)
        self.exploration_goal_noise = np.array([float(config["exploration_goal_noise"])] * self.goal_dim)

    def set_tensorboard_writer(self, writer):
        self.writer = writer

        # set writers in agents
        for i_level in range(self.k_level):
            self.HAC[i_level].set_tensorboard_writer(self.writer)

    def create_subgoal(self):
        # visualize ee position(subgoal)
        self.env.sim.create_sphere(
            body_name="ee_position", radius=0.02, mass=0, position=[0, 0, 0], rgba_color=[0.1, 0.1, 0.1, 0.3], ghost=True
        )

        # visualize subgoal_1
        self.env.sim.create_sphere(
            body_name="subgoal_1", radius=0.02, mass=0, position=[0, 0, 0], rgba_color=[0.9, 0.1, 0.1, 0.3], ghost=True
        )

        # visualize subgoal_2
        self.env.sim.create_sphere(
            body_name="subgoal_2", radius=0.02, mass=0, position=[0, 0, 0], rgba_color=[0.1, 0.9, 0.1, 0.3], ghost=True
        )

    def render_subgoal(self, subgoal):
        # render subgoals with setting new position.
        self.env.sim.set_base_pose("ee_position", subgoal[:3], [0, 0, 0, 1])
        self.env.sim.set_base_pose("subgoal_1", subgoal[4:7], [0, 0, 0, 1])
        self.env.sim.set_base_pose("subgoal_2", subgoal[7:], [0, 0, 0, 1])

    def check_goal(self, achieved_goal, desired_goal):
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return (d < self.threshold).astype(np.float32)

    def _ag(self, obs):
        """Return achieved goal from observation in block-gripper-informed-goal format"""
        ee_position = np.array(obs[: len(self.env.robot.get_ee_position())])
        fingers_width = np.array([obs[len(self.env.robot.get_obs())]])
        blocks = np.concatenate(
            [
                np.array(self.env.sim.get_base_position("object1")),
                np.array(self.env.sim.get_base_position("object2")),
            ]
        )
        ag = np.concatenate((ee_position, fingers_width, blocks))
        assert self.goal_dim == len(ag)
        return ag

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

                    if distance(action[4:7], action[7:]) < 0.02:  # object size/2
                        action[-1] *= 3  # if blocks are ovelapped stack-up

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True

                # Pass subgoal to lower level
                if self.render:
                    self.render_subgoal(action)
                next_state, done = self.run_HAC(i_level - 1, state, action, is_next_subgoal_test)

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.check_goal(self._ag(next_state["observation"]), action):
                    self.replay_buffer[i_level].add(
                        (
                            state,
                            action,
                            -self.H,
                            next_state["observation"],
                            goal,
                            0.0,
                            float(done),
                        )
                    )

                # for hindsight action transition
                action = self._ag(next_state["observation"])

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
                next_state, rew, _, info = self.env.step(action)
                done = info['is_success']

                # this is for logging
                self.reward += rew
                self.timestep += 1

            #   <================ finish one step/transition ================>
            # check if goal is achieved
            if i_level == self.k_level - 1:
                goal_achieved = self.check_goal(next_state["achieved_goal"], goal)
                if goal_achieved: 
                    print(f"level: {i_level} | goal_achieved! {goal_achieved}")
                    print(f"timestep: {self.timestep}")
                    print(next_state["achieved_goal"], goal)
            else:
                goal_achieved = self.check_goal(self._ag(next_state["observation"]), goal)
                if goal_achieved:
                    print(f"level: {i_level} | goal_achieved! {goal_achieved}")
                    print(f"timestep: {self.timestep}")
                    print(self._ag(next_state["observation"]), goal)


            # hindsight action transition
            if goal_achieved:
                self.replay_buffer[i_level].add(
                    (
                        state,
                        action,
                        0.0,
                        next_state["observation"],
                        goal,
                        0.0,
                        float(done),
                    )
                )
            else:
                self.replay_buffer[i_level].add(
                    (
                        state,
                        action,
                        -1.0,
                        next_state["observation"],
                        goal,
                        self.gamma,
                        float(done),
                    )
                )

            # copy for goal transition
            goal_transitions.append(
                [
                    state,
                    action,
                    -1.0,
                    next_state["observation"],
                    None,
                    self.gamma,
                    float(done),
                ]
            )

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
            if i_level == self.k_level - 1:
                transition[4] = next_state["achieved_goal"]
            else:
                transition[4] = self._ag(next_state["observation"])
            self.replay_buffer[i_level].add(tuple(transition))

        return next_state, done

    def train(self, max_episodes, save_episode, save_path=("", "")):

        self.timestep = 0
        # training procedure
        for i_episode in range(1, max_episodes + 1):
            self.reward = 0

            state = self.env.reset()
            # collecting experience in environment
            last_state, done = self.run_HAC(self.k_level - 1, state["observation"], state["desired_goal"], False)

            # update all levels
            self.update(self.n_iter, self.batch_size)

            if i_episode % save_episode == 0:
                self.save(save_path)

            self.writer.add_scalar("reward", self.reward, self.timestep)
            print("Episode: {}\t Reward: {}".format(i_episode, self.reward))

    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size, self.timestep)

    def save(self, save_path):
        directory, name = save_path
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + "_level_{}".format(i))

    def load(self, save_path):
        directory, name = save_path
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + "_level_{}".format(i))
