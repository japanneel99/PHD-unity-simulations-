from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def kmeans(data_, n_clusters):
    x = data_.iloc[:, 0:5]  # iloc allows us to pic a cell in the dataset

    kmeans = KMeans(n_clusters)
    kmeans.fit(x)

    identified_clusters = kmeans.fit_predict(x)

    centroids = kmeans.cluster_centers_

    data_with_clusters = data_.copy()
    data_with_clusters['Clusters'] = identified_clusters

    return data_with_clusters, identified_clusters, centroids


def plot_relative_distance(df_, clusters, centres, mode):
    if mode == 'relative_distance':
        plt.scatter(df_['relative_distance_index'], df_[
            'q_value'], c=clusters, cmap='rainbow')
        plt.title(
            'A Kmeans cluster plot of Q values vs Relative distance index feelings not considered')
        plt.xlabel('Relative distance index')
        plt.ylabel('Q Values')
        plt.savefig(
            "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/clustering_noPCA/no_feeling/relative_distance.jpg")
        plt.show()

    elif mode == 'relative_angle':
        plt.scatter(df_['relative_angle_index'], df_[
            'q_value'], c=clusters, cmap='rainbow')
        plt.title(
            'A Kmeans cluster plot of Q values vs Relative Angle index feelings not considered')
        plt.xlabel('Relative Angle index')
        plt.ylabel('Q Values')
        plt.savefig(
            "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/clustering_noPCA/no_feeling/relative_angle.jpg")
        plt.show()

    elif mode == 'relative_velocity':
        plt.scatter(df_['relative_velocity_index'], df_[
            'q_value'], c=clusters, cmap='rainbow')
        plt.title('A Kmeans cluster plot of Q values vs Relative Velocity index')
        plt.xlabel('Relative Velocity index')
        plt.ylabel('Q Values')
        plt.savefig(
            "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/clustering_noPCA/no_feeling/relative_velocity.jpg")
        plt.show()

    elif mode == "3d":
        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes(projection="3d")
        ax.scatter3D(df_['relative_distance_index'], df_['relative_angle_index'], df_[
                     'q_value'], c=clusters, cmap='rainbow')
        ax.set_title('3D KMeans plot of the Q values')
        ax.set_xlabel('relative_distance_index')
        ax.set_ylabel('relative_angle_index')
        ax.set_zlabel('Q value')
        ax.legend()
        plt.savefig(
            "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/clustering_noPCA/no_feeling/3D_Qclusters.jpg")
        plt.show()


def main():
    data = pd.read_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_tableNF.csv')

    number_of_clusters = 3

    mode = "3d"

    data_clusters, cluster_number, centroids = kmeans(data, number_of_clusters)

    plot_relative_distance(data_clusters, cluster_number, centroids, mode)


main()
