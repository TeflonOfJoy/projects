import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.neighbors import NearestNeighbors

def plot_digit(X, y, row_index):
    row = X.iloc[row_index, 0:28*28].to_numpy()
    label = y[row_index]
    fig, ax = plt.subplots()
    ax.imshow(row.reshape((28, 28)), cmap="binary")
    ax.set_title(f"Mnist digit {row_index} with label {label:.0f}")
    plt.show()

def plot_neighbors(Xtrain, ytrain, Xtest, ytest, indices, norm=None):
    labels = ytrain.to_numpy()[indices]

    columns = { f"label_neighbor{i}": labels[:,i] for i in range(labels.shape[1]) }
    columns["true_label"] = ytest

    # Neatly formatted print out of the nearest neighbors' labels
    print(pd.DataFrame(data=columns).astype(int))

    for i in range(indices.shape[0]):
        fig, ax = plt.subplots(1, indices.shape[1]+1)
        row = Xtest.iloc[i, 0:28*28].to_numpy()
        ax[-1].imshow(row.reshape((28, 28)), cmap="binary", norm=norm)
        ax[-1].set_title(f"Test example\ntrue label {ytest.iloc[i]:.0f}")
        for j in range(indices.shape[1]):
            row = Xtrain.iloc[indices[i, j], 0:28*28].to_numpy()
            ax[j].imshow(row.reshape((28, 28)), cmap="binary", norm=norm)
            ax[j].set_title(f"Neighbor {j}\ntrue label {ytrain.iloc[indices[i, j]]:.0f}")
        plt.show()

def train_knn(X, k=2):
    t = time.time()
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    print(f"KD-tree constructed in {time.time()-t:.2f}s")
    return knn

def evaluate_knn(knn, X, y):
    t = time.time()
    distances, indices = knn.kneighbors(X)
    print(f"Nearest neighbors found in {time.time()-t:.2f}s")
    return distances, indices

def effect_of_normalization(X):
    coeff = np.repeat([1, 1, 0.5, 0.01, 10, 0.1, 1], 4)
    coeffm = np.tile(coeff, 28).reshape((28, 28)).T
    coeffm *= coeff
    #plt.clf()
    #plt.imshow(coeffm, norm=LogNorm(vmin=0.01**2, vmax=10**2))
    #plt.title("Normalization coefficients")
    coeff = coeffm.ravel()
    return X * coeff

def effect_of_irrelevant_features(X, num, scale=100, seed=0):
    state = np.random.RandomState(seed)
    r = pd.DataFrame(state.random((X.shape[0], num)) * 255 * scale,
            columns=[f"rand{i}" for i in range(num)],
            index=X.index)
    return pd.concat((X, r), axis=1)

def main():
    X = pd.read_hdf("mnist.h5", key="X")
    y = pd.read_hdf("mnist.h5", key="y")

    # plot a random digit, for fun!
    plot_digit(X, y, 125)

    # train and test set
    perm = np.random.permutation(X.shape[0])
    m = int(X.shape[0]*0.8)
    Itrain = perm[0:m]
    Itest = perm[m:]
    Xtrain, ytrain = X.iloc[Itrain], y[Itrain]
    Xtest, ytest = X.iloc[Itest], y[Itest]

    print("QUESTION: Do the labels match?")
    print("          How would you use the neighbors for prediction?")
    knn = train_knn(Xtrain, k=3)

    # evaluate a small subset of 10 test set examples
    Xtest0, ytest0 = Xtest.iloc[0:10,:], ytest[0:10]
    distances, indices = evaluate_knn(knn, Xtest0, ytest0)
    plot_neighbors(Xtrain, ytrain, Xtest0, ytest0, indices)

    print("QUESTION: What is the effect of feature normalization?")
    print("          MNIST is by default normalized to values between 0-255,")
    print("          we rescale the values so that they are no longer normalized.")
    print("          Is any information lost?")
    Xtrain_norm = effect_of_normalization(Xtrain)
    Xtest0_norm = effect_of_normalization(Xtest0)
    knn = train_knn(Xtrain_norm, k=3)
    distances, indices = evaluate_knn(knn, Xtest0_norm, ytest0)
    labels = ytrain.to_numpy()[indices]
    plot_neighbors(Xtrain_norm, ytrain, Xtest0_norm, ytest0, indices,
            norm=LogNorm(0.01**2*255, 10**2*255))

    print("QUESTION: What is the effect of irrelevant features?")
    Xtrain_irr = effect_of_irrelevant_features(Xtrain, num=500, scale=1, seed=10)
    Xtest0_irr = effect_of_irrelevant_features(Xtest0, num=500, scale=1, seed=10)
    knn = train_knn(Xtrain_irr, k=3)
    distances, indices = evaluate_knn(knn, Xtest0_irr, ytest0)
    labels = ytrain.to_numpy()[indices]
    plot_neighbors(Xtrain_irr, ytrain, Xtest0_irr, ytest0, indices)

    print("QUESTION: Use sklearn.tree.DecisionTreeClassifier to learn a decision tree.")
    print("          What are the main drawbacks of k-NN?")

if __name__ == "__main__":
    main()


    
