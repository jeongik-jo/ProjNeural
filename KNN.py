from sklearn.neighbors import KNeighborsRegressor
import HyperParameters as hp
import numpy as np
import Dataset
from scipy.stats import iqr


def evaluate(X_train, y_train, X_test, y_test):
    n_neighbors_sizes = []
    mean_l2_errors = []
    median_l2_errors = []
    l2_iqrs = []

    for n_neighbors in hp.knn_neighbor_sizes:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        l2_errors = np.square(model.predict(X_test) - y_test)

        n_neighbors_sizes.append(n_neighbors)
        mean_l2_errors.append(np.mean(l2_errors))
        median_l2_errors.append(np.median(l2_errors))
        l2_iqrs.append(iqr(l2_errors))

    i = np.argmin(median_l2_errors)
    print('n neighbors :', n_neighbors_sizes[i])
    print('min mean l2 :', mean_l2_errors[i])
    print('min median l2 :', median_l2_errors[i])
    print("iqr :", l2_iqrs[i])


def main():
    (X_train, y_train), (X_test, y_test) = Dataset.load_dataset()
    evaluate(X_train, y_train, X_test, y_test)
main()