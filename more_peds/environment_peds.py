import numpy as np
import random
import math
import matplotlib as plt
from tensorforce.environments import Environment


class SimulatorEnvironment(Environment):
    def __init__(self, dt, W1, W2, W3, W4, a):
        self.dt = dt
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W4 = W4
        self.a = a

    def states(self):
        return dict(type='float', shape=(6,))

    def actions(self):
        return dict(type='float', min_value=0.0, max_value=1.6)

    def reset(self):
        self.pos_rx = 8.0  # + random.randint(8, -4)
        self.pos_ry = 0.0
        self.pos_rz = -6.0  # + random.randint(-4, 12)
        self.vel_rx = 0.0
        self.vel_ry = 0.0
        self.pos_p1x = 4.0  # + random.randint(-5, 6)
        self.pos_p1y = 0.0
        self.pos_p1z = 0.0  # + random.randint(-4, 6)
        self.vel_p1x = 1.4  # + random.randint(-0.4, 0.2)
        self.vel_p1y = 0.0
        self.vel_p1z = 0.0
        self.pos_p2x = 2.0  # + random.randint(-2, 8)
        self.pos_p2y = 0.0
        self.pos_p2z = -2.0  # + random.randint(-4, 10)
        self.vel_p2x = 1.2  # + random.randint(-0.2, 0.4)
        self.vel_p2y = 0.0
        self.vel_p2z = 0.0

        self.data = np.array([self.pos_rx, self.pos_ry, self.pos_rz, self.pos_p1x,
                             self.pos_p1y, self.pos_p1z, self.pos_p2x, self.pos_p2y, self.pos_p2z])

        self.relative_distance_p1 = math.sqrt(
            (self.data[2] - self.data[5])**2 + (self.data[0] - self.data[3])**2)

        self.relative_distance_p2 = math.sqrt(
            (self.data[2] - self.data[8])**2 + (self.data[0] - self.data[6])**2)

        self.relative_angle_p1 = math.atan2(
            (self.data[3] - self.data[0]), (self.data[5] - self.data[2]))
        self.relative_angle_p2 = math.atan2(
            (self.data[6] - self.data[0]), (self.data[8] - self.data[2]))

        self.relative_velocity_p1 = self.vel_p1x - self.vel_rx
        self.relative_velocity_p2 = self.vel_p2x - self.vel_rx

        self.state = [self.relative_distance_p1, self.relative_distance_p2, self.relative_angle_p1,
                      self.relative_angle_p2, self.relative_velocity_p1, self.relative_velocity_p2]

        return self.state

    def execute(self, actions):
        new_pos_rx = self.vel_rx * self.dt + self.pos_rx
        new_pos_ry = self.vel_ry * self.dt + self.pos_ry
        new_pos_rz = actions * self.dt + self.pos_rz
        new_pos_p1x = self.vel_p1x * self.dt + self.pos_p1x
        new_pos_p1y = self.vel_p1y * self.dt + self.pos_p1y
        new_pos_p1z = self.vel_p1z * self.dt + self.pos_p1z
        new_pos_p2x = self.vel_p2x * self.dt + self.pos_p2x
        new_pos_p2y = self.vel_p2y * self.dt + self.pos_p2y
        new_pos_p2z = self.vel_p2z * self.dt + self.pos_p2z

        feeling = 0

        r_p1 = math.sqrt((new_pos_rz - new_pos_p1z)**2 +
                         (new_pos_rx - new_pos_p1x)**2)
        r_p2 = math.sqrt((new_pos_rz - new_pos_p2z)**2 +
                         (new_pos_rx - new_pos_p2x)**2)

        terminal = new_pos_rz > new_pos_p1z + 8

        reward = (self.W2*actions) - self.W3*(math.exp(-(self.a*(r_p1**2)))
                                              ) - self.W4*(math.exp(-(self.a*(r_p2**2))))

        self.pos_rx = new_pos_rx
        self.pos_ry = new_pos_ry
        self.pos_rz = new_pos_rz
        self.pos_p1x = new_pos_p1x
        self.pos_p1y = new_pos_p1y
        self.pos_p1z = new_pos_p1z
        self.pos_p2x = new_pos_p2x
        self.pos_p2y = new_pos_p2y
        self.pos_p2z = new_pos_p2z

        self.next_data = np.array([self.pos_rx, self.pos_ry, self.pos_rz, self.pos_p1x,
                                  self.pos_p1y, self.pos_p1z, self.pos_p2x, self.pos_p2y, self.pos_p2z])

        self.relative_distance_p1 = math.sqrt(
            (self.next_data[2] - self.next_data[5])**2 + (self.next_data[0] - self.next_data[3])**2)

        self.relative_distance_p2 = math.sqrt(
            (self.next_data[2] - self.next_data[8])**2 + (self.next_data[0] - self.next_data[6])**2)

        self.relative_angle_p1 = math.atan2(
            (self.next_data[3] - self.next_data[0]), (self.next_data[5] - self.next_data[2]))
        self.relative_angle_p2 = math.atan2(
            (self.next_data[6] - self.next_data[0]), (self.next_data[8] - self.next_data[2]))

        self.relative_velocity_p1 = self.vel_p1x - self.vel_rx
        self.relative_velocity_p2 = self.vel_p2x - self.vel_rx

        self.next_state = [self.relative_distance_p1, self.relative_distance_p2, self.relative_angle_p1,
                           self.relative_angle_p2, self.relative_velocity_p1, self.relative_velocity_p2]

        return self.next_state, terminal, reward, feeling
