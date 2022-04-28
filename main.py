from pickle import FALSE
from tensorforce.agents import Agent
import numpy as np
from unity_environment import UnityEnvironment
from environment import SimulatorEnvironment
from qtable_agent import QtableAgent
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
    a = 0.2
    if is_unity:
        environment = UnityEnvironment(dt, W1, W2, W3, a)
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, a)
    return environment


def visualize_q_table(agent, environment):
    dt = 0.01

    states_list, actions_list, terminal_list, reward_list, feeling_list = simulate(
        environment, agent, True, custom_epsilon=0.0)  # ここではepsilonは常に0です。

    states_list = np.array(states_list)
    relative_distances = states_list[:, 0]
    relative_thetas = states_list[:, 1]
    relative_velocities = states_list[:, 2]
    time_step = np.linspace(0, dt*len(relative_distances),
                            num=len(relative_distances))

    rewards_episode = []
    episode = []

    for i in range(len(reward_list)):
        step = reward_list[i]
        x = i
        rewards_episode.append(step)
        episode.append(x)

    for i in range(10):
        print((i+1)*10, "mean_episode_rewards",
              np.mean(rewards_episode[10*i: 10*(i+1)]))

    rewards_per_10 = []
    episodes_10 = []

    for i in range(10):
        x_l = (i+1)*10
        y_l = np.mean(rewards_episode[10*i: 10*(i+1)])
        episodes_10.append(x_l)
        rewards_per_10.append(y_l)

    print("relative_distance", relative_distances)
    print("relative_angle", relative_thetas)

    plt.subplot(3, 1, 1)
    plt.title("epoch = 100 wx, wz = (10,-4) px,pz = (5,0)")
    plt.plot(relative_distances)
    plt.ylabel("relative distance")
    plt.xlabel("time step")
    plt.subplot(3, 1, 2)
    plt.plot(relative_thetas)
    plt.ylabel("relative angle")
    plt.xlabel("time step")
    plt.subplot(3, 1, 3)
    plt.plot(actions_list)
    plt.ylabel("Wheelchair z velocity")
    plt.xlabel("time step")
    plt.show()

    plt.subplot(3, 1, 1)
    plt.title("graphs")
    plt.plot(time_step, relative_distances)
    plt.ylabel("relative distance")
    plt.xlabel("time step")
    plt.subplot(3, 1, 2)
    plt.plot(time_step, relative_thetas)
    plt.ylabel("relative angle")
    plt.xlabel("time step")
    plt.subplot(3, 1, 3)
    plt.plot(time_step, actions_list)
    plt.ylabel("Wheelchair z velocity")
    plt.xlabel("time step")
    plt.show()

    plt.plot(time_step, reward_list)
    plt.ylabel("rewards")
    plt.xlabel("time_step")
    plt.show()

    plt.plot(episodes_10, rewards_per_10)
    plt.ylabel("Rewards_Episode_10")
    plt.xlabel("Episode_10")
    plt.xlim(0, 20)
    plt.show()

    # save all necessary data in a csv file to print graphs in matlab - add this stuff later
    agent.save(directory="saved_variables")

    # save the relevant data to a csv so i  can plot a nice graph in matlab.matlabではグラフを描けるようcsvにデータを保存
    if input("Save data in a csv file?(y/n)") == "y":
        with open('Tensorforce/Q_Learning/environments/everyStepQ/Expt41_300/Data/Expt41_300L.csv', 'w', newline='') as csvfile:
            fieldnames = ['Time_Step', 'Wheelchair_z_velocity',
                          'relative_distance', 'relative_angle', 'relative_velocity', 'feelings_episode', 'rewards']

            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writeheader()

            for i in range(len(relative_distances)):
                thewriter.writerow({'Time_Step': time_step[i], 'Wheelchair_z_velocity': actions_list[i], 'relative_distance':
                                   relative_distances[i], 'relative_angle': relative_thetas[i], 'relative_velocity': relative_velocities[i], 'feelings_episode': feeling_list[i], 'rewards': reward_list[i]})


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

        # print(states)
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
        print("%d th learning...." % i)

        states_list, actions_list, terminal_list, reward_list, feeling_list = \
            simulate(environment, agent, not use_experience,
                     None)

        states_learned.append(deepcopy(states_list))
        actions_learned.append(deepcopy(actions_list))
        terminal_learned.append(deepcopy(terminal_list))
        rewards_learned.append(deepcopy(reward_list))
        feeling_learned.append(deepcopy(feeling_list))

        if i % 100 == 0 and save_every_10:
            directory = "%d_model" % i
            if not os.path.exists(directory):
                os.mkdir(directory)
                agent.save(directory)

        if use_experience:
            agent.experience(
                states=states_list, actions=actions_list, terminal=terminal_list, reward=reward_list)
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

    # Save episodes and rewards per episode in a csv
    # if input("Save reward_data in csv file? (y/n)") == "y":
    with open('Tensorforce/Q_Learning/environments/everyStepQ/Expt41_300/Data/rewards_40300L.csv', 'w', newline='') as csvfile:
        fieldnames = ['episodes', 'ys']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        for i in range(len(ys)):
            thewriter.writerow({'episodes': episodes[i], 'ys': ys[i]})


def main():
    is_unity = input(
        "Use_unity_environment to do experiment?(y/n)").strip() == "y"
    environment = create_environment(is_unity)

    if input("Use qtable - Q learning?(y/n)").strip() == "n":
        agent = Agent.create(
            agent='tensorforce', environment=environment, update=64,
            optimizer=dict(optimizer='adam', learning_rate=1e-3),
            objective='policy_gradient', reward_estimation=dict(horizon=20)
        )
    elif input("load saved q-table from simulator?(y/n)").strip() == "n":
        agent = QtableAgent(
            action_candidates=np.linspace(0, 1.6, 17),
            quantization=[
                (0.4, 16.0, 10),
                (-math.pi, +math.pi, 6),
                (-1.4, 1.4, 2)],
            epsilon=0.1,
            alpha=0.1,
            gamma=0.9
        )
    else:
        n = int(input("which generation you want to load> ").strip())
        directory = "%d_model" % n
        agent = QtableAgent.load(directory)

    if input("only visualization? - This will make the graph smooth.(y/n)").strip() == "y":
        visualize_q_table(agent, environment)
        return

    # learn(environment, agent, n_epoch=101,
    #       save_every_10=True, use_experience=False)  # if i want to use experience then i need to change this to true

    states_learned, actions_learned, terminal_learned, rewards_learned, feeling_learned = learn(
        environment, agent, n_epoch=301, save_every_10=True, use_experience=False)

    agent.save("saved_variables")

    # import os
    # os.mkdir("a")
    # agent.save("a")
    # xs = np.load("a/q_table.npy")
    # print("now shape is ", xs.shape)

    # visualize latest agent
    visualize_q_table(agent, environment)

    if input("Save rewards in a CSV file? (y/n)") == "y":
        save_rewards(environment, agent, rewards_learned)


if __name__ == "__main__":
    main()
