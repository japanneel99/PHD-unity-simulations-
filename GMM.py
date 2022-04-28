from statistics import fmean
from sklearn.decomposition import PCA  # principle component analysis
from sklearn import mixture  # for Gaussian Mixture Model for classification
from sklearn import datasets
import itertools
from scipy import linalg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
sns.set(context="paper", style="whitegrid", rc={"figure.facecolor": "white"})

# FROM this code we can use the baysien information criterion BIC to get the number of components.


dataset = pd.read_csv(
    'Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Data/q_table.csv')
dataset.isnull().sum()

X = dataset.values
lowestBIC = np.infty
bic = []
nComponentsRange = range(1, 7)
cvTypes = ['spherical', 'tied', 'diag', 'full']
nInit = 10

for cvType in cvTypes:
    for nComponents in nComponentsRange:
        gmm = mixture.GaussianMixture(
            n_components=nComponents, covariance_type=cvType, n_init=nInit, init_params='kmeans')

        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowestBIC:
            lowestBIC = bic[-1]
            bestGmm = gmm


bic = np.array(bic)
colorIter = itertools.cycle(
    ['navy', 'turquoise', 'cornflowerblue', 'darkorange'])

bars = []

# plot the BIC Scores
plt.figure(figsize=(8, 6), dpi=100)
ax = plt.subplot(111)
for i, (cvType, color) in enumerate(zip(cvTypes, colorIter)):
    xpos = np.array(nComponentsRange) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(nComponentsRange): (i+1) * len(nComponentsRange)], width=.2, color=color))

plt.xticks(nComponentsRange)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(nComponentsRange)) + .65 + .2 * \
    np.floor(bic.argmin() / len(nComponentsRange))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
ax.set_xlabel('Number of Components')
ax.set_ylabel('BIC score')
ax.legend([b[0] for b in bars], cvTypes)
plt.savefig(
    "Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Graphs/Clustering/Feeling/BIC_Score.jpg")
plt.show()
