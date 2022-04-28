from matplotlib import projections
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def normalize_data(data_, n_comps):

    pca_ = PCA(n_comps)  # No of dimensions 次元数

    data_frame = pd.DataFrame(data_)
    #dataScaled = preprocessing.normalize(data_)
    dataScaled = preprocessing.normalize(data_frame)

    df_ = pca_.fit_transform(dataScaled)
    return df_


def Plot_Kmeans(df_, no_of_clus, mode):
    kmeans = KMeans(no_of_clus)

    label = kmeans.fit_predict(df_)
    # print(label)

    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    print(u_labels)
    print(centroids)

    if mode == "2d":
        for i in u_labels:
            plt.scatter(df_[label == i, 0], df_[label == i, 1], label=i)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig(
            "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/Clustering/feeling/2D_Kmeans.jpg")
        plt.show()

    elif mode == "3d":
        fig = plt.figure(figsize=(20, 16))
        ax = plt.axes(projection="3d")
        for i in u_labels:
            ax.scatter3D(df_[label == i, 0], df_[label == i, 1],
                         df_[label == i, 2], label == i)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   centroids[:, 2], s=80, color='k')
        ax.set_title('KMeans clustering 3D plot, human rewards not considered')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(['0', '1', '2'])
        ax.ylim(-1, 1)
        plt.savefig(
            "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/Clustering/no_feeling/3D_Kmeans.jpg")
        plt.show()


def main():
    data = pd.read_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_table.csv')

    num_components = 3
    num_clusters = 3
    varNames = data.columns

    mode = "2d"

    dataSet = normalize_data(data, num_components)
    Plot_Kmeans(dataSet, num_clusters, mode)


main()
