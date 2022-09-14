from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import Dataset
import time


def train(X_train, y_train, X_test, y_test):
    model = KernelReg(endog=y_train, exog=[X_train[:, i] for i in range(X_train.shape[1])],
                      var_type='c' * X_train.shape[1], reg_type='lc')
    return model


def validation(model, X_valid, y_valid):
    loss = np.mean((model.fit([X_valid[:, i] for i in range(X_valid.shape[1])])[0] - y_valid) ** 2)
    print('\ntest loss:\t', loss)
    return loss


def main(i):
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset(i)

    model = train(X_train, y_train, X_test, y_test)
    return validation(model, X_valid, y_valid)


if __name__ == "__main__":
    main(0)
