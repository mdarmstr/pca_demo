import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def pca(Xtrain, Xtest, ncomponents=2):
    pcaMdl = PCA(n_components=2) # Create a PCA model with 2 components
    TscrsTrain = pcaMdl.fit_transform(Xtrain)
    TscrsTest = pcaMdl.transform(Xtest)

    return TscrsTrain, TscrsTest


def plotPCA(TscrsTrain, TscrsTest, yTrain, yTest):
    plt.scatter(TscrsTrain[:, 0], TscrsTrain[:, 1], c=yTrain, alpha=0.75)
    plt.scatter(TscrsTest[:, 0], TscrsTest[:, 1], c=yTest, marker="*", alpha=0.75)
    plt.title("Principal Component Analysis")
    plt.ylabel("PC 1")
    plt.xlabel("PC 2")
    plt.show()


if __name__ == '__main__':
    coffeeDataX = pd.read_csv("X.csv", sep=",") # import X data
    coffeeDataY = pd.read_csv("y.csv", sep=",") # import Y data
    coffeeDataY = pd.get_dummies(coffeeDataY).to_numpy(dtype="int32").reshape(-1)

    # Scale the data: SNV followed by mean-centring
    xSNV = scale(coffeeDataX, axis=1) # SNV scaling
    xtrain, xtest, ytrain, ytest = train_test_split(xSNV, coffeeDataY, test_size=0.3, random_state=42)
    meanCentre = StandardScaler(with_mean=True, with_std=False)
    meanCentre.fit_transform(xtrain)
    meanCentre.transform(xtest)

    TscrsTrain, TscrsTest = pca(xtrain, xtest)
    plotPCA(TscrsTrain, TscrsTest, ytrain, ytest)

