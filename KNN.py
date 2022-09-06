from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import Dataset


neighbor_sizes = [1, 2, 3, 4, 8, 12, 16, 20]


def train(X_train, y_train, X_test, y_test):
    min_loss = np.inf
    min_model = None
    min_n_neighbors = None

    for n_neighbors in neighbor_sizes:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        test_loss = np.mean(np.square(model.predict(X_test) - y_test))

        if test_loss < min_loss:
            min_loss = test_loss
            min_model = model
            min_n_neighbors = n_neighbors

    print('valid loss:', min_loss)
    print('neighbors:', min_n_neighbors)

    return min_model


def validation(model, X_valid, y_valid):
    loss = np.mean(np.square(model.predict(X_valid) - y_valid))
    print('\ntest loss:\t', loss)


def main():
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset()
    model = train(X_train, y_train, X_test, y_test)
    validation(model, X_valid, y_valid)


main()
