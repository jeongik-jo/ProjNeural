from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import Dataset
import time

bandwidths = [2 ** i for i in range(-5, 6)]


def train(X_train, y_train, X_test, y_test):
    min_loss = np.inf
    min_model = None
    min_bandwidth = None

    for bandwidth in bandwidths:
        model = KernelReg(endog=y_train, exog=[X_train[:, i] for i in range(X_train.shape[1])],
                          var_type='c' * X_train.shape[1], reg_type='lc', bw=np.full([X_train.shape[1]], bandwidth))
        test_loss = np.mean(np.square(model.fit([X_test[:, i] for i in range(X_test.shape[1])])[0] - y_test))

        if test_loss < min_loss:
            min_loss = test_loss
            min_model = model
            min_bandwidth = bandwidth

    print('test loss:', min_loss)
    print('bandwidth:', min_bandwidth)

    return min_model


def validation(model, X_valid, y_valid):
    loss = np.mean((model.fit([X_valid[:, i] for i in range(X_valid.shape[1])])[0] - y_valid) ** 2)
    print('\nvalid loss:\t', loss)
    return loss


def main(i):
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset(i)

    model = train(X_train, y_train, X_test, y_test)
    return validation(model, X_valid, y_valid)


if __name__ == "__main__":
    main(0)
