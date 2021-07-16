import RobotDART as Rd
import numpy as np
from math import cos, sin, degrees
from machin.frame.algorithms import PPO
import torch
from retry import retry
import pandas as pd


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 4)
        self.fc2 = torch.nn.Linear(4, 4)

        self.mu_head = torch.nn.Linear(4, action_num)
        self.sigma_head = torch.nn.Linear(4, action_num)

        self.action_range = 4

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
        # self.graphics.look_at([1, 0, 1])

        # load pendulum
        self.pendulum = Rd.Robot("pendulum.urdf")
        self.pendulum.fix_to_world()
        self.pendulum.set_actuator_types("torque")

        # set its initial position as the lowest possible
        positions = self.pendulum.positions()
        positions[0] = 0
        self.pendulum.set_positions(positions)

        # add robot to the simulation
        self.simu.add_robot(self.pendulum)

        self.pendulum.set_commands(np.array([3]))

    def step(self, velocity):
        self.pendulum.set_commands(velocity)

        for _ in range(20):
            if self.simu.step_world():
                break

        theta = self.pendulum.positions()[0]

        reward = 5 * cos(theta) - 20 * np.sign(sin(theta)) * np.sign(self.pendulum.velocities()[0])
        next_state = (self.pendulum.positions()[0], self.pendulum.velocities()[0])
        done = False

        return next_state, reward, done

    # reset the simulation
    def reset(self):
        positions = self.pendulum.positions()
        positions[0] = np.pi
        self.pendulum.set_external_torque(self.pendulum.body_name(1), [0., 0., 0.])
        self.pendulum.set_positions(positions)

        starting_state = (positions[0], self.pendulum.velocities()[0])

        return starting_state

    def enable_graphics(self):
        self.simu.set_graphics(self.graphics)
        self.graphics.look_at([1, 0, 1])


@retry(Exception, tries=3, delay=0, backoff=0)
def ppo_sim(print_flag, max_episodes, max_steps):
    # configurations
    pendulum_simulation = Simulation()
    state_dim = 2
    action_dim = 1
    action_range = 4
    # max_episodes = 500
    # max_steps = 100
    solved_reward = 500
    solved_repeat = 15
    data = []  # for each episode's data
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    ppo = PPO(actor,
              critic,
              torch.optim.Adam,
              torch.nn.MSELoss(reduction='sum'),
              actor_learning_rate=0.0001,
              critic_learning_rate=0.0001,
              batch_size=50
              )

    reward_fulfilled = 0
    smoothed_total_reward = 0.0

    m_angle = 0

    for episode in range(1, max_episodes + 1):
        episode_reward = 0.0
        terminal = False
        step = 0
        state = torch.tensor(pendulum_simulation.reset(), dtype=torch.float32).view(1, state_dim)

        tmp_observations = []
        angle = 0

        while not terminal and step < max_steps:
            step += 1

            with torch.no_grad():
                # Observe the state of the environment and take action
                # Take random actions in the first 20 episodes
                if episode < 20:
                    action = ((2.0 * torch.rand(1, 1) - 1.0) * action_range)
                    torque = action
                else:
                    action = ppo.act({"state": state})[0]
                    torque = action[0]

                next_state, reward, terminal = pendulum_simulation.step(torque)
                next_state = torch.tensor(next_state, dtype=torch.float32).view(1, state_dim)
                episode_reward += reward

                tmp_observations.append({
                    "state": {"state": state},
                    "action": {"action": action},
                    "next_state": {"state": next_state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps,
                }
                )

                state = next_state

            current_angle = degrees(pendulum_simulation.pendulum.positions()[0]) % 360
            current_angle = current_angle if current_angle <= 180 else 360 - current_angle
            angle += current_angle / max_steps

        ppo.store_episode(tmp_observations)  # this should be here. Not after the first 20 episodes.

        if episode > max_episodes - 100:
            m_angle += angle / 100

        if print_flag:
            print(f"Episode: [{episode:3d}/{max_episodes:3d}] Reward: {episode_reward:.2f} Angle: {angle:.2f}",
                  end="\r")
            print("", end="\n")
        else:
            data_curr = [episode, episode_reward, angle]
            data.append(data_curr)
        smoothed_total_reward = smoothed_total_reward * 0.9 + episode_reward * 0.1

        if episode > 20:
            ppo.update()

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1

            if reward_fulfilled >= solved_repeat:
                print("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    if print_flag:
        print(m_angle)
    else:
        data_df = pd.DataFrame(data, columns=['Episode', 'Reward', 'Angle'])
        return data_df


if __name__ == '__main__':
    ppo_sim(1, 200, 200)
