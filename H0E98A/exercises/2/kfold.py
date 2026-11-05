import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_moons

def main():
    #loss = "binary_crossentropy" # was removed in version 1.3
    loss = "log_loss"

    # Generate the training and test data
    rand = np.random.RandomState(31)
    X, y = make_moons(1000, noise=0.3, random_state=rand)
    Xtest, ytest = make_moons(1000, noise=0.3, random_state=rand)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, marker=".")
    ax.set_title("The data")
    plt.show()

    # We have a number of possible hyper parameters for our classifier
    # We want to determine the best hyper parameter set
    settings = [
            {"learning_rate": 0.5, "num_trees": 10, "max_leaf_nodes": 7},
            {"learning_rate": 0.5, "num_trees": 10, "max_leaf_nodes": 15},
            {"learning_rate": 0.2, "num_trees": 20, "max_leaf_nodes": 31},
            {"learning_rate": 0.4, "num_trees": 20, "max_leaf_nodes": 31},
            {"learning_rate": 0.2, "num_trees": 60, "max_leaf_nodes": 63},
            {"learning_rate": 0.4, "num_trees": 60, "max_leaf_nodes": 63}]

    # K-fold cross validation
    # To find the best parameter set, we train and evaluate the model on
    # different subsets of the training data. This gives us an a good idea of
    # how the model performs.
    K = 10
    perm = rand.permutation(X.shape[0])
    fold_size = perm.shape[0] / K
    folds = [perm[int(i*fold_size):int((i+1)*fold_size)] for i in range(K)]

    accuracies_train = np.zeros((K, len(settings)))
    accuracies_val = np.zeros((K, len(settings)))

    for i in range(K): # leave one fold as validation set
        Itrain = np.hstack([folds[j] for j in range(K) if i!=j])
        Xtrain = X[Itrain, :]
        ytrain = y[Itrain]
        Ival = folds[i]
        Xval = X[Ival, :]
        yval = y[Ival]

        print(f"Training fold {i}")
        for j, s in enumerate(settings):
            print(f"- {s} -> ", end="")
            clf = HistGradientBoostingClassifier(
                    loss=loss,
                    max_iter=s["num_trees"],
                    max_leaf_nodes=s["max_leaf_nodes"],
                    learning_rate=s["learning_rate"],
                    random_state=rand)
            clf.fit(Xtrain, ytrain)
            acc_train = np.mean(clf.predict(Xtrain) == ytrain)
            acc_val = np.mean(clf.predict(Xval) == yval)

            print(f"accuracy train: {acc_train*100:.1f}%, test: {acc_val*100:.1f}%")
            accuracies_train[i, j] = acc_train
            accuracies_val[i, j] = acc_val

    print("Mean accuracies train:", accuracies_train.mean(axis=0).round(3))
    print("            std train:", accuracies_train.std(axis=0).round(3))
    print("Mean accuracies validation:", accuracies_val.mean(axis=0).round(3))
    print("            std validation:", accuracies_val.std(axis=0).round(3))

    print("QUESTION: Which model do you think performs best?")
    print("          Which model would you pick based on the training accuracy?")
    print("          Which model would you pick based on the validation accuracy?")

    # We pick the first and the last configuration, and train a model on all
    # training data, and plot the decision surfaces.
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
    fig_test, ax_test = plt.subplots(1, 2, figsize=(15, 8))
    for i in [0, -1]:
        s = settings[i]
        clf = HistGradientBoostingClassifier(
                loss=loss,
                max_iter=s["num_trees"],
                max_leaf_nodes=s["max_leaf_nodes"],
                learning_rate=s["learning_rate"],
                random_state=rand)
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        acc_test = np.mean(clf.predict(Xtest) == ytest)

        print(f"Accuracy train: {acc*100:.1f}% test: {acc_test*100:.1f}%, "
              f"gap={(acc-acc_test)*100:.1f}%")

        plot_step = 0.01
        xmin, xmax = X[:, 0].min()-0.1, X[:, 0].max()+0.1
        ymin, ymax = X[:, 1].min()-0.1, X[:, 1].max()+0.1
        xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step),
                             np.arange(ymin, ymax, plot_step))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cs = ax[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        cs = ax_test[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        ax[i].scatter(X[:, 0], X[:, 1], c=y, marker=".")
        ax_test[i].scatter(Xtest[:, 0], Xtest[:, 1], c=ytest, marker=".")
        conf_name = "Simple model" if i==0 else "Complex model"
        fig.suptitle("Scatter points are training data")
        ax[i].set_title(f"{conf_name} accuracy train: {acc*100:.1f}% test: {acc_test*100:.1f}%")
        fig_test.suptitle("Scatter points are testing data")
        ax_test[i].set_title(f"{conf_name} accuracy train: {acc*100:.1f}% test: {acc_test*100:.1f}%")
    plt.show()

    print("QUESTION: Looking at the decision boundaries of the two models,")
    print("          which of the two would you prefer and why?")


    print()
    print("QUESTION: Use `clf.predict_proba` to predict probabilities instead")
    print("          of classes and use these probability values to compute the")
    print("          values in a confusion matrix for an arbitrary threshold.")
    print("          Then, repeat this for all possible thresholds and plot the")
    print("          ROC curve.")



if __name__ == "__main__":
    main()
