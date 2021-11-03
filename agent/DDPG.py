import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, action_bounds, offset):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        # max value of actions
        self.action_bounds = action_bounds
        self.offset = offset

    def forward(self, state, goal):
        with torch.no_grad():
            input_ = torch.cat([state, goal], 1)
            action = (self.actor(input_).detach().cpu().data.numpy() * self.action_bounds) + self.offset
        return torch.FloatTensor(action).to(device)


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, H):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.H = H

    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        return -self.critic(torch.cat([state, action, goal], 1)) * self.H


class DDPG:
    def __init__(self, state_dim, goal_dim, action_dim, action_bounds, offset, lr, H, tau, name=""):
        self.name = name

        self.actor = Actor(state_dim, goal_dim, action_dim, action_bounds, offset).to(device)
        self.target_actor = Actor(state_dim, goal_dim, action_dim, action_bounds, offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, goal_dim, action_dim, H).to(device)
        self.target_critic = Critic(state_dim, goal_dim, action_dim, H).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()
        
        self.tau = tau
        self.soft_update(1.0)

    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()

    def set_tensorboard_writer(self, writer):
        self.writer = writer

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, buffer, n_iter, batch_size, timestep):

        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size, 1)).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # select next action
            next_action = self.target_actor(next_state, goal).detach()

            # Compute target Q-value:
            target_Q = self.target_critic(next_state, next_action, goal).detach()
            target_Q = reward + ((1 - done) * gamma * target_Q)

            # Compute critic loss
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.writer.add_scalar(f"{self.name}/critic_loss", critic_loss, timestep)
            # Optimize Critic:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
            self.writer.add_scalar(f"{self.name}/actor_loss", actor_loss, timestep)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft-update target networks
            self.soft_update()

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, name), map_location="cpu"))
