from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import Dataset
from scipy.stats import iqr


neighbor_sizes = [1, 2, 3, 4, 8, 12, 16, 20]


def train(X_train, y_train, X_valid, y_valid):
    min_loss = np.inf
    min_model = None
    min_n_neighbors = None

    for n_neighbors in neighbor_sizes:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        loss = np.mean(np.square(model.predict(X_valid) - y_valid))

        if loss < min_loss:
            min_loss = loss
            min_model = model
            min_n_neighbors = n_neighbors

    print('valid loss:', min_loss)
    print('neighbors:', min_n_neighbors)

    return min_model


def test(model, X_test, y_test):
    losses = np.square(model.predict(X_test) - y_test)
    print('\nmse:\t', np.mean(losses))
    print('median:\t', np.median(losses))
    print("iqr:\t", iqr(losses))


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = Dataset.load_dataset()
    model = train(X_train, y_train, X_valid, y_valid)
    test(model, X_test, y_test)

main()