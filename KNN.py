from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import Dataset
import time

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

    print('test loss:', min_loss)
    print('neighbors:', min_n_neighbors)

    return min_model


def validation(model, X_valid, y_valid):
    loss = np.mean(np.square(model.predict(X_valid) - y_valid))
    print('\nvalid loss:\t', loss)
    return loss


def main(i):
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset(i)

    start = time.time()
    model = train(X_train, y_train, X_test, y_test)
    print('train time: ', time.time() - start, '\n')

    return validation(model, X_valid, y_valid)


if __name__ == "__main__":
    main(0)
