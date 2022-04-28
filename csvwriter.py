
from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pandas import DataFrame
from sklearn import datasets
from mpl_toolkits import mplot3d


def save_csv():
    q_table = np.load(
        f"Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Q_tables/300_model/q_table.npy")

    q_r = []
    r_d = []
    r_t = []
    r_v = []
    a = []

    for rd in range(10):
        for rt in range(6):
            for rv in range(2):
                for action in range(17):
                    q = q_table[rd, rt, rv, action]
                    q_r.append([rd, rt, rv, action, q])

    print(q_table)
    print(q_r)

    df = pd.DataFrame(q_r, columns=[
        "relative_distance_index", "relative_angle_index", "relative_velocity_index", "action_index", "q_value"])

    df.to_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_table.csv', index=False)


def plot_graph():
    Data1D = pd.read_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_table.csv')

    x = Data1D['relative_distance_index'].values
    y = Data1D['relative_angle_index'].values
    z = Data1D['q_value'].values
    a = Data1D['relative_velocity_index'].values
    acc = Data1D['action_index'].values

    x_1 = np.array(x)
    y_1 = np.array(y)
    #y_1 = np.array(a)
    z_1 = np.array(z)
    a_mesh = np.array(acc)

    # create a 2d x, y grid (both x and y will be 2D)
    X, Y = np.meshgrid(x_1, y_1)

    # repeat Z to make it a 2d grid
    Z = np.tile(z_1, (len(z_1), 1))

    fig = plt.figure(figsize=(12, 10))
    #ax3d = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.plot_surface(X, Y, Z, color='orangered', edgecolors='yellow')
    # ax.set_title(
    #     'Surface plot of the Q values - Human rewards are considered', fontsize=24)
    # ax.set_xlabel('Relative distance index', fontsize=24)
    # ax.set_ylabel('Relative angle index', fontsize=24)
    # ax.set_zlabel('Q value', fontsize=26)
    ax.set_zlim(-60, 80)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    plt.show()


def main():
    if input("Save values in a csv for 3D plotting and GMM? (y/n)").strip() == "y":
        save_csv()

    if input("Plot a surface plot to visualize the Q values, surface and mesh plot? (y/n)").strip() == "y":
        plot_graph()
        plt.show()


if __name__ == "__main__":
    main()
