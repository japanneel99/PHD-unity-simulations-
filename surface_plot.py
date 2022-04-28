import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot_graph():
    Data1D = pd.read_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_tableNF.csv')

    x = Data1D['relative_distance'].values
    y = Data1D['relative_angle'].values
    z = Data1D['q_value'].values
    a = Data1D['relative_velocity'].values

    x_1 = np.array(x)
    y_1 = np.array(y)
    #y_1 = np.array(a)
    z_1 = np.array(z)

    # create a 2d x, y grid (both x and y will be 2D)
    #X, Y = np.meshgrid(x_1, y_1)
    X, Y = np.meshgrid(x_1, y_1)

    # repeat Z to make it a 2d grid
    Z = np.tile(z_1, (len(z_1), 1))

    fig = plt.figure()
    #ax3d = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_xlabel('Relative distance index', fontsize=30)
    ax.set_ylabel('Relative angle index', fontsize=30)
    ax.set_zlabel('Q value', fontsize=30)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='z', labelsize=22)
    plt.show()


def main():
    if input("Plot a surface plot to visualize the Q values? (y/n)").strip() == "y":
        plot_graph()
        plt.show()


if __name__ == "__main__":
    main()
