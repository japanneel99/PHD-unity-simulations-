from telnetlib import SB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from IPython.display import Image

# Load the dataset
df = pd.read_csv(
    'Tensorforce/Q_Learning/environments/everyStepQ/Expt28_100/Data/q_table.csv')
df.isnull().sum()
print(df.head())

# Explore the data statistics
print(df.describe())

my_dataframe = df[df.q_value.isnull() == False]

print(my_dataframe.describe())

# Now i will split the data into training and test set to train
y = my_dataframe.q_value
X = my_dataframe.drop('q_value', axis=1)
# scaling data
#X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# scaling data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)  # or: fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


if input("plot with histogram? (y/n)").strip() == "y":
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.set_palette('rainbow')
    sns_plot = sns.pairplot(df, hue="action_index",
                            diag_kind="kde", height=2.0)
    sns_plot.savefig(
        "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/GMM/Hist.png")
    plt.clf()
    Image(filename='Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/GMM/Hist.png')

else:
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns_plot = sns.pairplot(df, hue="action_index", height=2.0)
    sns_plot.savefig(
        "Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/GMM/no_hist.png")
    plt.clf()
    Image(filename='Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Graphs/GMM/no_hist.png')
