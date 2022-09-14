import numpy as np
from itertools import product
import Dataset
import time

A = 1
N = 2
R = 10 ** 6
r = 4
Ms = [2, 4, 8, 16]

I_n = 400
d = Dataset.input_dim
s = np.ceil(np.log2(N + 1))  # 2
c_3 = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f_id(x):
    return 4 * R * sigmoid(x / R) - 2 * R


def f_mult(x, y):
    return (R ** 2) / 4 * (1 + np.exp(-1.0)) ** 3 / (np.exp(-2.0) - np.exp(-1.0)) * \
           (sigmoid(2 * (x + y) / R + 1) - 2 * sigmoid((x + y) / R + 1)
            - sigmoid(2 * (x - y) / R + 1) + 2 * sigmoid((x - y) / R + 1))


def f_relu(x):
    return f_mult(f_id(x), sigmoid(R * x))


def get_B(X, M, b):
    u = -np.sqrt(d) * A + np.arange(0, M + 1) * 2 * np.sqrt(d) * A / M

    def f_hat(x, y):
        return f_relu(M / (2 * np.sqrt(d) * A) * (x - y) + 1) - 2 * f_relu(M / (2 * np.sqrt(d) * A) * (x - y)) + \
               f_relu(M / (2 * np.sqrt(d) * A) * (x - y) - 1)

    B = []
    for l in np.arange(1, r + 1):
        for k in np.arange(1, M + 2):
            for j in product(np.arange(0, N + 1), repeat=d):
                if 0 <= np.sum(j) <= N:
                    def f_down_up(x, down, up):
                        if 0 <= up <= s - 1:
                            return f_mult(f_down_up(x, 2 * down - 1, up + 1), f_down_up(x, 2 * down, up + 1))
                        else:
                            for t in np.arange(1, d + 1):
                                if np.sum(j[:t - 1]) + 1 <= down <= np.sum(j[:t]):
                                    return f_id(f_id(x[:, t - 1]))
                            if down == np.sum(j) + 1:
                                return f_hat(x @ b[l - 1], u[k - 1])
                            else:
                                return np.ones(shape=[x.shape[0]])
                    B.append(f_down_up(X, 1, 0))
    B = np.array(B).T

    return B


def predict(X, M, a, b):
    return get_B(X, M, b) @ a


def train_step(X_train, y_train, M, prev_loss, prev_a, prev_b):
    b = np.random.uniform(-1, 1, size=[r, d])
    B = get_B(X_train, M, b)
    a = np.linalg.inv(B.T @ B + np.eye(B.shape[1]) * c_3) @ (B.T @ y_train)

    loss = np.mean(np.square(predict(X_train, M, a, b) - y_train))

    if loss < prev_loss:
        return loss, a, b
    else:
        return prev_loss, prev_a, prev_b


def train(X_train, y_train, X_test, y_test):
    min_loss = np.inf
    min_a = None
    min_b = None
    min_M = None

    for M in Ms:
        train_loss = np.inf
        a = None
        b = None
        for _ in range(I_n):
            train_loss, a, b = train_step(X_train, y_train, M, train_loss, a, b)

        test_loss = np.mean((predict(X_test, M, a, b) - y_test) ** 2)
        if test_loss < min_loss:
            min_loss = test_loss
            min_M = M
            min_a = a
            min_b = b

    print('test loss:\t', min_loss)
    print('min M:\t', min_M)
    return min_M, min_a, min_b


def validation(X_valid, y_valid, M, a, b):
    loss = np.mean((predict(X_valid, M, a, b) - y_valid) ** 2)
    print('valid loss:\t', loss)
    return loss


def main(i):
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset(i)
    start = time.time()
    M, a, b = train(X_train, y_train, X_test, y_test)
    print('train time:\t', time.time() - start)
    return validation(X_valid, y_valid, M, a, b)


if __name__ == "__main__":
    main(0)