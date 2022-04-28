import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
sns.set(context="paper", style="whitegrid", rc={"figure.facecolor": "white"})

"""
Primrary component analysis 
"""


df = pd.read_csv(
    'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_table.csv')

df.isnull().sum()

varNames = df.columns

# normalization
dfScaled = df.copy()[varNames]
dfScaled = (dfScaled - dfScaled.mean()) / \
    dfScaled.std()  # std is the standard deviation

# K-means clustering
nClusters = 3
nInit = 30
randomState = 4

pred = KMeans(n_clusters=nClusters, init='k-means++', n_init=nInit,
              random_state=randomState).fit_predict(dfScaled[varNames])
dfScaled["clusterId"] = pred


# Principal Components Analysis
pca = PCA(n_components=5)
trans = pca.fit_transform(dfScaled[varNames])


def summaryPCA(pca_, cols):
    # Contribution rate of each dimension
    print('各次元の寄与率: {0}'.format(pca_.explained_variance_ratio_))
    # Cumulative contribution rate
    print('累積寄与率: {0}'.format(sum(pca_.explained_variance_ratio_)))
    print("PCA Components")
    for k, v in zip(cols, pca.components_.T):
        print(k, ":", ", ".join(["%2.2f" % i for i in v]))

    for i, ratio in enumerate(np.concatenate((np.array([0]), np.cumsum(pca_.explained_variance_ratio_)))):
        print('{}'.format(i), 'components:',
              '{:.2f}'.format(ratio*100), '% are explained')
    plt.plot(list(range(6)), np.concatenate(
        (np.array([0]), np.cumsum(pca_.explained_variance_ratio_))), 'o-')
    plt.xlabel('# of components')
    plt.ylabel('Explained variance ratio')
    plt.title('A graph showing the explained variance ratio vs number of components')
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/Variance_ratio.jpg")
    plt.show()

    plt.figure(figsize=(10, 10))
    var = np.round(pca_.explained_variance_ratio_ * 100, decimals=1)
    lbls = [str(x) for x in range(1, len(var) + 1)]
    plt.bar(x=range(1, len(var) + 1), height=var, tick_label=lbls)
    plt.title('A  bar graph illustrating the components ratio')
    plt.xlabel('Component')
    plt.ylabel('The Contribution of each component')
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/bar_variance.jpg")
    plt.show()


def contriPCAPlot(pca_, cols):
    c_p = pca_.components_[:2]

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim([np.min(c_p[0] + [0]) - 0.1, np.max(c_p[0]) + 0.1])
    ax.set_ylim([np.min(c_p[1]+[0])-0.1, np.max(c_p[1])+0.1])
    # colormap setting

    cm = plt.get_cmap("hsv")
    c = []
    n_ = len(c_p[0])
    for i in range(n_):
        c.append(cm(i/n_))

    for i, v in enumerate(c_p.T):
        ax.arrow(x=0, y=0, dx=v[0], dy=v[1], width=0.01, head_width=0.05,
                 head_length=0.05, length_includes_head=True, color=c[i])
    # legend setting
    patch = []
    for i in range(n_):
        patch.append(mpatches.Patch(color=c[i], label=cols[i]))
    plt.legend(handles=patch, bbox_to_anchor=(1.05, 1),
               loc='upper left', borderaxespad=0, fontsize=8)
    ax.set_title("pca_components_for each variable")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/2DArrowPCA.jpg")
    plt.show()


def clusterPCAPlot(trans_, df_):
    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    for i in range(len(df_["clusterId"].unique())):
        ax.scatter(trans_[df_["clusterId"] == i, 0],
                   trans_[df_["clusterId"] == i, 1])
    plt.legend(df_["clusterId"].unique())
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('A PCA cluster graph')
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/2DPCA_cluster.jpg")
    plt.show()

################################## From here i will plot 3d graphs for clustering #########################


def clusterPCA3dPlot(trans_, df_):
    fig = plt.figure(figsize=(6, 6), dpi=150)
    for k in range(1, 5):
        ax = fig.add_subplot(2, 2, k, projection='3d')
        for i in range(len(df_["clusterId"].unique())):
            cond = df_["clusterId"] == i
            ax.scatter(trans_[cond, 0], trans_[cond, 1], trans_[cond, 2])

        ax.view_init(elev=20., azim=(k-1)*30)
        ax.set_title("PCA 3d plot")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        if k in (2, 4):
            ax.legend(df_["clusterId"].unique())
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/3DPCA_cluster.jpg")
    plt.tight_layout()
    plt.show()


def contriPCAPlot3d(pca_, cols):
    c_p = pca_.components_[:3]

    fig = plt.figure(figsize=(6, 6), dpi=150)
    for k in range(1, 5):
        ax = fig.add_subplot(2, 2, k, projection='3d')
        ax.set_xlim([np.min(c_p[0] + [0]) - 0.1, np.max(c_p[0]) + 0.1])
        ax.set_ylim([np.min(c_p[1] + [0]) - 0.1, np.max(c_p[1]) + 0.1])
        ax.set_zlim([np.min(c_p[2] + [0]) - 0.1, np.max(c_p[2]) + 0.1])
        # color setting
        cm = plt.get_cmap("hsv")
        c = []
        n_ = len(c_p[0])
        for i in range(n_):
            c.append(cm(i/n_))

        for i, v in enumerate(c_p.T):
            # quiver plot is used to plot arrows
            ax.quiver(0, 0, 0, v[0], v[1], v[2], lw=1, color=c[i])

        if k in range(2, 4):
            # legend setting
            patch = []
            for i in range(n_):
                patch.append(mpatches.Patch(color=c[i], label=cols[i]))
            ax.legend(handles=patch, bbox_to_anchor=(1.05, 1),
                      loc='upper left', borderaxespad=0, fontsize=8)

        ax.set_title("Pca components for each variable")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.view_init(elev=20, azim=(k-1)*30)
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/3DArrow_PCA.jpg")
    plt.show()


def clusterPCAContri3dPlot(pca_, df_, trans_, cols):
    c_p = pca_.components_[:3]

    fig = plt.figure(figsize=(8, 12), dpi=150)
    for k in range(1, 5):
        # Plot a scatter plot here of what is above
        ax = fig.add_subplot(4, 2, 2*k-1, projection='3d')
        for i in range(len(df_['clusterId'].unique())):
            cond = df_["clusterId"] == i
            ax.scatter(trans_[cond, 0], trans_[cond, 1], trans_[cond, 2])

        ax.view_init(elev=20., azim=(k-1)*30)  # because nInit is set to 30
        ax.set_title('PCA 3D plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(df_['clusterId'].unique())

        ##contribution arrow part########
        ax = fig.add_subplot(4, 2, 2*k, projection="3d")
        ax.set_xlim([np.min(c_p[0] + [0]) - 0.1, np.max(c_p[0]) + 0.1])
        ax.set_ylim([np.min(c_p[1] + [0]) - 0.1, np.max(c_p[1]) + 0.1])
        ax.set_zlim([np.min(c_p[2] + [0]) - 0.1, np.max(c_p[2]) + 0.1])
        # set the color
        cm = plt.get_cmap("hsv")
        c = []
        n_ = len(c_p[0])
        for i in range(n_):
            c.append(cm(i/n_))

        for i, v in enumerate(c_p.T):
            ax.quiver(0, 0, 0, v[0], v[1], v[2], lw=1, color=c[i])

        # legend
        patch = []
        for i in range(n_):
            patch.append(mpatches.Patch(color=c[i], label=cols[i]))
        ax.legend(handles=patch, bbox_to_anchor=(0.95, 1),
                  loc='upper left', borderaxespad=0, fontsize=8)

        ax.set_title('PCA components for each variable')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.view_init(elev=20., azim=(k-1)*30)
    plt.tight_layout()
    # plt.savefig(
    #     "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/PCA/Feeling/3D_clusters_arrows.jpg")
    plt.show()


def main():

    summaryPCA(pca, varNames)

    contriPCAPlot(pca, varNames)

    clusterPCAPlot(trans, dfScaled)

    clusterPCA3dPlot(trans, dfScaled)

    contriPCAPlot3d(pca, varNames)

    clusterPCAContri3dPlot(pca, dfScaled, trans, varNames)


main()
