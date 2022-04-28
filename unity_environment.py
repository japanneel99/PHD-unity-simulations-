import io
from os import stat_result
import socket
import time
import json
from matplotlib.pyplot import title, xcorr, ylabel
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import csv
import random


class UnityEnvironment:
    def __init__(self, dt, W1, W2, W3, a):
        host = "127.0.0.1"
        port = 25001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.dt = dt
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.a = a

    def states(self):
        return dict(type='float', shape=(12,))

    def get_state_from_unity(self):
        self.sock.sendall("GET_WHEELCHAIR_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        wheelchair_position_dict = json.loads(receivedData)
        wheelchair_position = [wheelchair_position_dict['x'],
                               wheelchair_position_dict['y'], wheelchair_position_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian_pos_dict = json.loads(receivedData)
        pedestrian_position = [pedestrian_pos_dict['x'],
                               pedestrian_pos_dict['y'], pedestrian_pos_dict['z']]

        self.sock.sendall("GET_WHEELCHAIR_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        wheelchair_vel_dict = json.loads(receivedData)
        wheelchair_velocity = [wheelchair_vel_dict['x'],
                               wheelchair_vel_dict['y'], wheelchair_vel_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian_vel_dict = json.loads(receivedData)
        pedestrian_velocity = [pedestrian_vel_dict['x'],
                               pedestrian_vel_dict['y'], pedestrian_vel_dict['z']]

        self.sock.sendall("GET_FEELING_REWARD".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        feeling = json.loads(receivedData)['info']
        print(receivedData)
        print("feeling recieved")

        return wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling

    def actions(self):
        return dict(type='float', num_values=17)

    def reset(self):
        print("resetting")
        self.sock.sendall("RESET".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling = self.get_state_from_unity()
        self.data = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                              wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                              pedestrian_position[0], pedestrian_position[1], pedestrian_position[2],
                              pedestrian_velocity[0], pedestrian_velocity[1], pedestrian_velocity[2]])

        relative_distance = math.sqrt(
            (self.data[2] - self.data[8]) ** 2 + (self.data[0] - self.data[6]) ** 2)
        relative_angle = math.atan2(
            (self.data[6] - self.data[0]), (self.data[8] - self.data[2]))
        relative_velocity = self.data[9] - self.data[3]

        self.state = [relative_distance, relative_angle, relative_velocity]

        print("Ended reset")
        return self.state

    def execute(self, actions):
        print("start executing")
        wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling = self.get_state_from_unity()

        terminal = wheelchair_position[2] > pedestrian_position[2] + \
            2 or pedestrian_position[1] > wheelchair_position[1] + 2

        r = math.sqrt((wheelchair_position[2] - pedestrian_position[2]) ** 2 + (
            wheelchair_position[0] - pedestrian_position[0]) ** 2)

        reward = (self.W1 * feeling) + (self.W2 * actions) - \
            self.W3*(math.exp(-(0.2*(r**2))))

        # reward = (self.W2 * actions) - self.W3*(math.exp(-(0.2*(r**2))))

        msg = "SEND_ACTION,%f" % actions
        self.sock.sendall(msg.encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        self.data = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                              wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                              pedestrian_position[0], pedestrian_position[1], pedestrian_position[2],
                              pedestrian_velocity[0], pedestrian_velocity[1], pedestrian_velocity[2]])

        relative_distance = math.sqrt(
            (self.data[2] - self.data[8]) ** 2 + (self.data[0] - self.data[6]) ** 2)
        numerator = self.data[0] - self.data[6]
        denominator = self.data[2] - self.data[8]
        relative_angle = math.atan2(numerator, denominator)

        relative_velocity = self.data[9] - self.data[3]

        self.state = [relative_distance, relative_angle, relative_velocity]

        print("execute done")
        return self.state, terminal, reward, feeling
