import numpy as np
import Dataset
import time
import pyearth


def train(X_train, y_train, X_test, y_test):
    min_loss = np.inf
    min_model = None
    min_kernel_name = None

    for _ in range(5):
        model = pyearth.Earth()
        model.fit(X_train, y_train)

        test_loss = np.mean(np.square(model.predict(X_test) - y_test))

        if test_loss < min_loss:
            min_loss = test_loss
            min_model = model

    print('test loss:', min_loss)
    print('kernel_name:', min_kernel_name)

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
