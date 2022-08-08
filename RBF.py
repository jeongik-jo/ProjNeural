from scipy.interpolate import RBFInterpolator
import HyperParameters as hp
import numpy as np
import Dataset


def evaluate(X_train, y_train, X_test, y_test):
    n_neighbors_sizes = []
    mean_l2_errors = []

    for n_neighbors in hp.knn_neighbor_sizes:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        mean_l2_error = np.mean(np.square(model.predict(X_test) - y_test))

        n_neighbors_sizes.append(n_neighbors)
        mean_l2_errors.append(mean_l2_error)

    i = np.argmin(mean_l2_errors)
    print('min l2 :', mean_l2_errors[i])
    print('n neighbors :', n_neighbors_sizes[i])


def main():
    (X_train, y_train), (X_test, y_test) = Dataset.load_dataset()
    evaluate(X_train, y_train, X_test, y_test)
main()