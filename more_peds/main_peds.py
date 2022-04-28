from pickle import FALSE
from tensorforce.agents import Agent
import numpy as np
from unity_environment_peds import UnityEnvironment
from environment_peds import SimulatorEnvironment
from qtable_agent_peds import QtableAgent
from copy import deepcopy
import matplotlib.pyplot as plt
import math
import os
import csv as csv
import pandas as pd


def create_environment(is_unity):
    dt = 0.01
    W1 = (10*1/3)
    W2 = (1/5 * 1/1.6)
    W3 = 5
    W4 = 5
    a = 0.2  # 行動ではなくガウス分布のa
    if is_unity:
        environment = UnityEnvironment(dt, W1, W2, W3, W4, a)
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, W4, a)
    return environment


def visualize_q_table(agent, environment):
    dt = 0.01

    states_list, actions_list, terminal_list, reward_list, feeling_list = simulate(
        environment, agent, True, custom_epsilon=0.0)

    states_list = np.array(states_list)
    relative_distances_p1 = states_list[:, 0]
    relative_distances_p2 = states_list[:, 1]
    relative_angles_p1 = states_list[:, 2]
    relative_angles_p2 = states_list[:, 3]
    relative_velocities_p1 = states_list[:, 4]
    relative_velocities_p2 = states_list[:, 5]
    time_step = np.linspace(
        0, dt*len(relative_distances_p1), num=len(relative_distances_p1))

    rewards_episode = []
    episodes = []

    print("relative_distances_p1", relative_distances_p1)
    print("relative_angles_p1", relative_angles_p1)
    print("relative_distances_p2", relative_distances_p2)
    print("relative_angles_p2", relative_angles_p2)

    plt.subplot(3, 1, 1)
    plt.plot(time_step, relative_distances_p1)
    plt.plot(time_step, relative_distances_p2)
    plt.subplot(3, 1, 2)
    plt.plot(time_step, relative_angles_p1)
    plt.plot(time_step, relative_angles_p2)
    plt.subplot(3, 1, 3)
    plt.plot(time_step, actions_list)
    plt.show()


def simulate(environment, agent, learn, custom_epsilon=None):
    states_list = []
    actions_list = []
    terminal_list = []
    reward_list = []
    feeling_list = []

    states = environment.reset()
    terminal = False
    while not terminal:

        if isinstance(agent, QtableAgent):
            actions = agent.act(states=states, custom_epsilon=custom_epsilon)
        else:
            actions = agent.act(states=states)

        states, terminal, reward, feeling = environment.execute(
            actions=actions)

        if learn:
            agent.observe(terminal=terminal, reward=reward)

        states_list.append(deepcopy(states))
        actions_list.append(deepcopy(actions))
        terminal_list.append(deepcopy(terminal))
        reward_list.append(deepcopy(reward))
        feeling_list.append(deepcopy(feeling))

    return states_list, actions_list, terminal_list, reward_list, feeling_list


def learn(environment, agent, n_epoch, save_every_10, use_experience):
    states_learned = []
    actions_learned = []
    terminal_learned = []
    rewards_learned = []
    feeling_learned = []

    for i in range(n_epoch):
        print("%d th learning....." % i)

        states_list, actions_list, terminal_list, reward_list, feeling_list = simulate(
            environment, agent, not use_experience, None)

        states_learned.append(deepcopy(states_list))
        actions_learned.append(deepcopy(actions_list))
        terminal_learned.append(deepcopy(terminal_list))
        rewards_learned.append(deepcopy(reward_list))
        feeling_learned.append(deepcopy(feeling_list))

        if i % 100 == 0 and save_every_10:
            directory = "%d_model" % i
            if not os.path.exists(directory):
                os.mkdir(directory)
                # agent.save(directory)

        if use_experience:
            agent.experience(states=states_list, actions=actions_list,
                             terminal=terminal_list, reward=reward_list)
            agent.update()

    return states_learned, actions_learned, terminal_learned, rewards_learned, feeling_learned


def save_rewards(environment, agent, rewards_learned):
    rewards_list_history = rewards_learned
    ys = []
    for reward in rewards_list_history:
        ys.append(np.average(reward))
    print("ys", ys)
    print(len(ys))

    episodes = []
    for i in range(len(ys)):
        step = i
        episodes.append(step)

    with open('/reward_test.csv', 'w', newline='') as csvfile:
        fieldnames = ['episodes', 'ys']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        for i in range(len(ys)):
            thewriter.writerow({'episodes': episodes[i], 'ys': ys[i]})


def main():
    is_unity = input(
        "Use_unity_environment to do experiment?(y/n)").strip() == "y"
    environment = create_environment(is_unity)

    if input("Use q-table Q-Learning?(y/n)").strip == "n":
        agent = Agent.create(agent='tensorfoce', environment=environment, update=64, optimizer=dict(
            optimizer='adam', learning_rate=1e-3), objective='policy_gradient', reward_estimation=dict(horizon=20))

    elif input("load saved q-table from simulator?(y/n)") == "n":
        agent = QtableAgent(action_candidates=np.linspace(0, 1.6, 17),
                            quantization=[(0.4, 16.0, 10),
                                          (-math.pi, +math.pi, 6),
                                          (-1.4, 1.4, 2)],
                            epsilon=0.1,
                            alpha=0.1,
                            gamma=0.9)
    else:
        n = int(input("which geenration do you want to load").strip())
        directory = "%d_model" % n
        agent = QtableAgent.load(directory)

    if input("only visualization? - This will make the graph smooth.(y/n)").strip() == "y":
        visualize_q_table(agent, environment)
        return

    states_learned, actions_learned, terminal_learned, rewards_learned, feeling_learned = learn(
        environment, agent, n_epoch=101, save_every_10=True, use_experience=False)

    # agent.save("saved_variables")

    visualize_q_table(agent, environment)

    if input("save rewards in a CSV file? (y/n)") == "y":
        save_rewards(environment, agent, rewards_learned)


if __name__ == "__main__":
    main()
