from scipy.interpolate import RBFInterpolator
import numpy as np
import Dataset
import time

kernel_names = ['cubic', 'linear', 'quintic', 'thin_plate_spline']


def train(X_train, y_train, X_test, y_test):
    min_loss = np.inf
    min_model = None
    min_kernel_name = None

    for kernel_name in kernel_names:
        model = RBFInterpolator(X_train, y_train, kernel=kernel_name)
        test_loss = np.mean(np.square(model(X_test) - y_test))

        if test_loss < min_loss:
            min_loss = test_loss
            min_model = model
            min_kernel_name = kernel_name

    print('test loss:', min_loss)
    print('kernel_name:', min_kernel_name)

    return min_model


def validation(model, X_valid, y_valid):
    loss = np.mean(np.square(model(X_valid) - y_valid))
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
