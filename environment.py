import numpy as np
import random
import math
import matplotlib as plt
from tensorforce.environments import Environment


class SimulatorEnvironment(Environment):
    def __init__(self, dt, W1, W2, W3, a):
        self.dt = dt
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.a = a

    def states(self):
        return dict(type='float', shape=(3,))

    def actions(self):
        return dict(type='float', min_value=0.0, max_value=1.6)

    def reset(self):
        self.pos_rx = 8.0 + random.randint(-8, 4)
        self.pos_ry = 0.0
        self.pos_rz = -6.0 + random.randint(-4, 12)
        self.vel_rx = 0.0
        self.vel_ry = 0.0
        #self.vel_rz = np.random.uniform(0, 1.6)
        self.pos_px = 4.0 + random.randint(-5, 6)
        self.pos_py = 0.0
        self.pos_pz = 0.0 + random.randint(-4, 6)
        self.vel_px = 1.4 + random.uniform(-0.4, 0.2)
        self.vel_py = 0.0
        self.vel_pz = 0.0

        self.data = np.array(
            [self.pos_rx, self.pos_ry, self.pos_rz, self.pos_px, self.pos_py, self.pos_pz])

        self.relative_distance = math.sqrt(
            (self.data[2] - self.data[5])**2 + (self.data[0] - self.data[3])**2)
        self.relative_angle = math.atan2(
            (self.data[3]-self.data[0]), (self.data[5] - self.data[2]))
        self.relative_velocity = self.vel_px - self.vel_rx

        self.state = [self.relative_distance,
                      self.relative_angle, self.relative_velocity]

        return self.state

    def execute(self, actions):
        new_pos_rx = self.vel_rx * self.dt + self.pos_rx
        new_pos_ry = self.vel_ry * self.dt + self.pos_ry
        new_pos_rz = actions * self.dt + self.pos_rz
        new_pos_px = self.vel_px * self.dt + self.pos_px
        new_pos_py = self.vel_py * self.dt + self.pos_py
        new_pos_pz = self.vel_pz * self.dt + self.pos_pz

        feeling = 0

        r = math.sqrt((new_pos_rz - new_pos_pz)**2 +
                      (new_pos_rx - new_pos_px)**2)

        #terminal_value = new_pos_pz + 8
        terminal = new_pos_rz > new_pos_pz + 8 or new_pos_px > new_pos_rx + 2

        reward = (self.W2*actions) - self.W3*(math.exp(-(self.a*(r**2))))

        self.pos_rx = new_pos_rx
        self.pos_ry = new_pos_ry
        self.pos_rz = new_pos_rz
        self.pos_px = new_pos_px
        self.pos_py = new_pos_py
        self.pos_pz = new_pos_pz

        self.next_data = np.array(
            [self.pos_rx, self.pos_ry, self.pos_rz, self.pos_px, self.pos_py, self.pos_pz])

        self.relative_distance = math.sqrt(
            (self.next_data[2] - self.next_data[5])**2 + (self.next_data[0] - self.next_data[3])**2)
        self.relative_angle = math.atan2(
            (self.next_data[3]-self.next_data[0]), (self.next_data[5] - self.next_data[2]))
        self.relative_velocity = self.vel_px - self.vel_rx

        self.next_state = [self.relative_distance,
                           self.relative_angle, self.relative_velocity]

        return self.next_state, terminal, reward, feeling
