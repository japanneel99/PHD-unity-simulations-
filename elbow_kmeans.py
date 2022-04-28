# K-means elbow method example
##################### code explanation ##############################################
"""
We need to come up with a method to help us decide how many clusters we should use for our K-means model
The ELBOW METHOD is a very popular technique, and the idea is to run k-means clustering for a range of clusters
k (e.g 1-7) and for each value, we are calculating the sum of squared distances from each point to its 
assuigned centre(distortions).
When the distortions are plotted and the plot looks like an arm, then the elbow(point of inflextion on the curve)
is the best value of k
K-meansモデルに使用するクラスターの数を決定する為にelbow_plot方法を適用する。
k-meansを何回(range)実行し、更に値ごとに所定された中心からの二乗距離を計算する。
distortionを描き、変曲点は最適なkである

"""
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


def elbow_plot(df, K):

    distortions = []

    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k - number of clusters')
    plt.ylabel('Distortions')
    plt.title(
        'The Elbow Method showing the optimal k where feelings are not considered')
    plt.savefig(
        "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/Clustering/no_feeling/elbow_plot.jpg")
    plt.show()


def main():

    data = pd.read_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_tableNF.csv')
    K = range(1, 10)  # Number of clusters.

    elbow_plot(data, K)


main()
