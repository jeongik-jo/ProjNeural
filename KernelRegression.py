from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import Dataset
import time


repeat_time = 10

def train(X_train, y_train, X_test, y_test):
    model = KernelReg(y_train, X_train, var_type='c', reg_type='lc')

    return model


def validation(model, X_valid, y_valid):
    loss = np.mean((model(X_valid) - y_valid) ** 2)
    print('\ntest loss:\t', loss.numpy())


def main(i):
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset(i)

    model = train(X_train, y_train, X_test, y_test)
    validation(model, X_valid, y_valid)


if __name__ == "__main__":
    main(0)
