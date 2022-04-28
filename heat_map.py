import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math

# np.clip anthing less than 0 will becomes 0 and anything > n_quad - 1 will be n_quad - 1
# if statement in quantization is upgrading to multi-dimensional numpy array


def quantization(x: Union[float, np.ndarray], x_min: float, x_max: float, n_quad: int) -> Union[int, np.ndarray]:
    index = ((x - x_min) / (x_max - x_min) * n_quad)
    if isinstance(index, np.ndarray):
        retval = index.astype(int)
        return np.clip(retval, 0, n_quad - 1)
    else:
        retval = int(index)
        return max(0, min(n_quad - 1, retval))


def create_v_table(q_table, velocities, mode="expected_v"):
    """creating v table from the saved q_table

    :param q_table: q_table[relative_distance_index, relative_angle_index, relative_velocity_index, action_index]
    :param velocities: velocities[action_index] is the actual velocity corresponding to the action_index
    :param mode:  "expected_v" or "maximum_v"
    :return: v_table[relative_distance_index, relative_angle_index, n_velocity]
    """

    n_distance = q_table.shape[0]  # 0th column shape, can do this with np
    n_angle = q_table.shape[1]  # 1th column shape
    n_velocity = q_table.shape[2]  # 2th column shape

    v_table = np.zeros((n_distance, n_angle, n_velocity))
    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                total = np.sum(q_table[i_distance, i_angle, i_velocity, :])
                if total == 0:
                    v_table[i_distance, i_angle, i_velocity] == -1
                elif mode == "expected_v":
                    probabilities = q_table[i_distance,
                                            i_angle, i_velocity, :] / total

                    # calculate the expected velocity
                    v_table[i_distance, i_angle, i_velocity] = np.sum(
                        probabilities * velocities)
                elif mode == "max_v":
                    i = np.argmax(q_table[i_distance, i_angle, i_velocity, :])
                    v_table[i_distance, i_angle, i_velocity] = velocities[i]
    return v_table


def get_expected_v(x, y, velocity_index, v_table):
    """get expected velocity at a specific state/position 

    :param x: relative x position 
    :param y: relative y position
    :param velocity_index: velocity_index 
    :param v_table: v_table 
    :return: This function will return the expected velocity at position (x, y) and velocity_index 
    """

    distance = np.sqrt(x**2, y**2)
    distance_index = quantization(distance, 0.4, 16, v_table.shape[0])
    angle = np.arctan2(y, x)
    angle_index = quantization(angle, -math.pi, math.pi, v_table.shape[1])

    return v_table[distance_index, angle_index, velocity_index]


def plot_v_table(v_table, velocity_index, min_v, max_v, n_resolution=500):
    # create the mesh we want to plot here
    relative_xs = np.linspace(-10, 10, num=n_resolution)
    relative_ys = np.linspace(-10, 10, num=n_resolution)
    mesh_relative_xs, mesh_relative_ys = np.meshgrid(relative_xs, relative_ys)
    z_mesh = get_expected_v(
        mesh_relative_xs, mesh_relative_ys, velocity_index, v_table)

    # plot the graph
    plt.pcolormesh(relative_xs, relative_ys, z_mesh,
                   shading="auto", vmin=min_v, vmax=max_v)
    plt.xlabel("relative x position", fontsize=30)
    plt.ylabel("relative y position", fontsize=30)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.rc('xtick', labelsize=26)
    plt.rc('ytick', labelsize=26)
    # plt.title(
    #     'A heat map showing the velocity at a given position, feelings considered', fontsize=30)
    plt.colorbar()
    plt.show()


def main():
    #q_table = np.load("q_table.npy")
    q_table = np.load(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Q_tablesNF/300_model/q_table.npy')
    # These are the actions the robot can take.
    velocities = np.linspace(0, 1.6, num=17)
    mode = "max_v"

    v_table = create_v_table(q_table, velocities, mode=mode)

    plot_v_table(v_table, velocity_index=1, min_v=-1, max_v=1.6)


if __name__ == "__main__":
    main()
