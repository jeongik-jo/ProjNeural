from scipy.interpolate import RBFInterpolator
import numpy as np
import Dataset

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


def main():
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset()
    model = train(X_train, y_train, X_test, y_test)
    validation(model, X_valid, y_valid)


main()
