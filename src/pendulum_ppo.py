import RobotDART as Rd
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        a = torch.relu(self.fc1(state_action))
        a = torch.relu(self.fc2(a))

        return self.fc3(a)


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))

        return a * self.action_range


class Simulation:
    def __init__(self):
        timestep = 0.004
        self.simu = Rd.RobotDARTSimu(timestep)
        self.simu.set_collision_detector("fcl")

        gconfig = Rd.gui.GraphicsConfiguration(1024, 768)
        graphics = Rd.gui.Graphics(gconfig)
        self.simu.set_graphics(graphics)
        graphics.look_at([1, 2.5, 2.5])

        # load pendulum
        pendulum = Rd.Robot("pendulum.urdf")
        pendulum.fix_to_world()
        pendulum.set_actuator_types("torque")

        # raise it to 1m so that it is able to rotate without hitting the ground
        raised_body_pose = pendulum.body_pose("base_link")
        raised_body_pose.set_translation([0, 0, 1])
        pendulum.set_base_pose(raised_body_pose)

        # set its initial position as the lowest possible
        positions = pendulum.positions()
        positions[0] = 3.141593
        pendulum.set_positions(positions)

        # add robot to the simulation
        self.simu.add_robot(pendulum)

        # add a floor
        self.simu.add_floor()

    def step(self):
        next_state = None
        reward = 0
        done = False

        return next_state, reward, done


if __name__ == "__main__":
    pendulum_simulation = Simulation()

    while True:
        if pendulum_simulation.simu.step_world():
            break
