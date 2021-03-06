import RobotDART as Rd
import numpy as np
from machin.frame.algorithms import PPO
from math import sqrt
import torch
from retry import retry
import pandas as pd


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 32)
        self.fc2 = torch.nn.Linear(32, 16)

        self.mu_head = torch.nn.Linear(16, action_num)
        self.sigma_head = torch.nn.Linear(16, action_num)

        self.action_range = 2

    def forward(self, state, action=None):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))

        mu = self.mu_head(a)
        sigma = torch.nn.functional.softplus(self.sigma_head(a))

        dist = torch.distributions.Normal(mu, sigma)
        act = action if action is not None else dist.rsample()
        act_entropy = dist.entropy()

        # the suggested way to confine your actions within a valid range
        # is not clamping, but remapping the distribution
        act_log_prob = dist.log_prob(act)
        act_tanh = torch.tanh(act)
        act = act_tanh * self.action_range

        # the distribution remapping process used in the original essay.
        act_log_prob -= torch.log(self.action_range * (1 - act_tanh.pow(2)) + 1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)

        return act, act_log_prob, act_entropy


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, state):
        v = torch.relu(self.fc1(state))
        v = torch.relu(self.fc2(v))
        v = self.fc3(v)

        return v


class Simulation:
    def __init__(self):
        timestep = 0.004
        self.simu = Rd.RobotDARTSimu(timestep)
        self.simu.set_collision_detector("fcl")

        gconfig = Rd.gui.GraphicsConfiguration(1024, 768)
        self.graphics = Rd.gui.Graphics(gconfig)
        # self.simu.set_graphics(self.graphics)
        # self.graphics.look_at([3., 1., 2.], [0., 0., 0.])

        # load iiwa
        packages = [("iiwa_description", "iiwa/iiwa_description")]
        self.iiwa = Rd.Robot("iiwa/iiwa.urdf", packages)
        self.iiwa.fix_to_world()
        self.iiwa.set_actuator_types("servo")

        positions = self.iiwa.positions()
        self.iiwa.set_positions(positions)

        robot_ghost = self.iiwa.clone_ghost()

        initial_positions = [1, 0.8, -1.12, -1.47, 1.02, -0.45, 0.91]
        self.iiwa.set_positions(initial_positions)

        # add robot to the simulation
        self.simu.add_robot(self.iiwa)
        self.simu.add_robot(robot_ghost)
        self.simu.add_floor()

    def step(self, positions):
        self.iiwa.set_commands(positions)

        for _ in range(20):
            if self.simu.step_world():
                break

        next_state = (self.iiwa.positions())

        reward = 0
        for pos in next_state:
            reward += pos * pos
        reward = - sqrt(reward)

        done = False

        return next_state, reward, done

    # reset the simulation
    def reset(self):
        starting_state = [1, 0.8, -1.12, -1.47, 1.02, -0.45, 0.91]
        self.iiwa.set_positions(starting_state)

        return starting_state


@retry(Exception, tries=5, delay=0, backoff=0)
def ppo_sim(print_flag, max_episodes, max_steps):
    iiwa_simulation = Simulation()
    state_dim = 7
    action_dim = 7
    action_range = 2
    # max_episodes = 500
    # max_steps = 200
    solved_reward = 500
    solved_repeat = 500
    data = []  # to get each episode's data

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    ppo = PPO(actor,
              critic,
              torch.optim.Adam,
              torch.nn.MSELoss(reduction='sum'),
              actor_learning_rate=0.0001,
              critic_learning_rate=0.0001,
              batch_size=20
              )

    reward_fulfilled = 0
    smoothed_total_reward = 0.0

    for episode in range(1, max_episodes + 1):
        episode_reward = 0.0
        terminal = False
        step = 0
        state = torch.tensor(iiwa_simulation.reset(), dtype=torch.float32).view(1, state_dim)

        tmp_observations = []

        while not terminal and step < max_steps:
            step += 1

            with torch.no_grad():
                # Observe the state of the environment and take action
                # Take random actions in the first 20 episodes
                if episode < 20:
                    action = ((2.0 * torch.rand(1, 7) - 1.0) * action_range)
                    torque = np.transpose(action)
                else:
                    action = ppo.act({"state": state})[0]
                    torque = np.transpose(action)

                next_state, reward, terminal = iiwa_simulation.step(torque)
                next_state = torch.tensor(next_state, dtype=torch.float32).view(1, state_dim)
                episode_reward += reward

                tmp_observations.append({
                    "state": {"state": state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps,
                }
                )

                state = next_state

        if print_flag:
            print(f"Episode: [{episode:3d}/{max_episodes:3d}] Reward: {episode_reward:.2f}", end="\r")
            print("", end="\n")
        else:
            data_curr = [episode, episode_reward]
            data.append(data_curr)

        smoothed_total_reward = smoothed_total_reward * 0.9 + episode_reward * 0.1

        ppo.store_episode(tmp_observations)

        if episode > 20:
            ppo.update()

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1

            if reward_fulfilled >= solved_repeat:
                print("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    if not print_flag:
        data_df = pd.DataFrame(data, columns=['Episode', 'Reward'])
        return data_df


if __name__ == '__main__':
    ppo_sim(1, 100, 200)
