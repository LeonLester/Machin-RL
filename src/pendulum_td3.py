import RobotDART as Rd
import numpy as np
from math import cos, sin, degrees
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch
from machin.frame.algorithms import TD3
import torch.nn as nn


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)

        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)

        self.action_range = action_range

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
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = torch.cat((state, action), 1)
        q = torch.relu(self.fc1(state_action))
        q = torch.relu(self.fc2(q))
        q = self.fc3(q)

        return q


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
        # 3 part reward function
        # 1.
        # 2. reduce reward the further that the next state's mechanical energy is from
        # being only potential energy which means that it should be as inverted as possible and as motionless as
        # possible
        # 3. increase reward as the last state's mechanical energy is the same. 2 and 3 ensure promote that
        # we want it to get close to inverted and still, without promoting getting away from it

        # reward = 25 * np.exp(-1 * (next_state[0] - 1) * (next_state[0] - 1) / 0.001) \
        #          - 100 * np.abs(10 * 0.5 - (10 * 0.5 * next_state[0] + 0.5 * 0.3333 * next_state[1] * next_state[1])) \
        #          + 100 * np.abs(10 * 0.5 - (10 * 0.5 * state[0, 0] + 0.5 * 0.3333 * state[0, 1] * state[0, 1]))
        # reward = reward.item()
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


if __name__ == '__main__':
    # configurations
    pendulum_simulation = Simulation()
    state_dim = 2
    action_dim = 1
    action_range = 4
    max_episodes = 500
    max_steps = 200
    solved_reward = 500
    solved_repeat = 15

    # td3 specific
    observe_dim = 2
    noise_param = (0, 0.2)
    noise_mode = "normal"

    actor = Actor(state_dim, action_dim, action_range)
    actor_t = Actor(state_dim, action_dim, action_range)
    critic = Critic(state_dim, action_dim)
    critic_t = Critic(state_dim, action_dim)
    critic2 = Critic(state_dim, action_dim)
    critic2_t = Critic(state_dim, action_dim)

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        torch.optim.Adam,
        nn.MSELoss(reduction="sum"),
        batch_size=100,
        actor_learning_rate=0.00001,
        critic_learning_rate=0.00001
    )

    episode, step, reward_fulfilled = 0, 0, 0
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
                    action = td3.act_with_noise({"state": state}, noise_param=noise_param, mode=noise_mode)[0]
                    torque = action[0]

                next_state, reward, terminal = pendulum_simulation.step(torque)
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

            current_angle = degrees(pendulum_simulation.pendulum.positions()[0]) % 360  # get the angle of the pendulum
            current_angle = current_angle if current_angle <= 180 else 360 - current_angle  # get the distance in radians from the goal [0,180)
            angle += current_angle / max_steps

        if episode > max_episodes - 100:
            m_angle += angle / 100  # get an average from the last 100 episodes

        print(f"Episode: [{episode:3d}/{max_episodes:3d}] Reward: {episode_reward:.2f} Angle: {angle:.2f}", end="\r")

        print("", end="\n")
        smoothed_total_reward = smoothed_total_reward * 0.9 + episode_reward * 0.1
        td3.store_episode(tmp_observations)
        if episode > 20:

        td3.store_episode(tmp_observations)

        if episode > 20:
            td3.update()

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1

            if reward_fulfilled >= solved_repeat:
                print("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    print(m_angle)
