import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import HyperParameters as hp
from itertools import product
import Dataset

A = 1
N = 2
R = 10**6
r = 4
Ms = [2]

I_n = 1
d = hp.input_dim
s = np.ceil(np.log2(N + 1)) # 2
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


(X_train, y_train), (X_test, y_test) = Dataset.load_dataset()

for M in Ms:
    min_loss = np.inf
    for _ in range(I_n):
        u = -np.sqrt(d) * A + np.arange(0, M + 1) * 2 * np.sqrt(d) * A / M

        def f_hat(x, y):
            return f_relu(M / (2 * np.sqrt(d) * A) * (x - y) + 1) - 2 * f_relu(M / (2 * np.sqrt(d) * A) * (x - y)) + \
                   f_relu(M / (2 * np.sqrt(d) * A) * (x - y) - 1)

        b = np.random.uniform(-1, 1, size=[r, d])
        B = []
        for l in np.arange(1, r + 1):
            for k in np.arange(1, M + 2):
                for j in product(np.arange(0, N + 1), repeat=d):
                    def f_down_up(x, down, up):
                        if (0 <= up <= s-1) and (1 <= down <= 2 ** up):
                            return f_mult(f_down_up(x, 2 * down - 1, up + 1), f_down_up(x, 2 * down, up + 1))
                        if up == s:
                            for t in np.arange(1, d + 1):
                                if np.sum(j[:t-1]) + 1 <= down <= np.sum(j[:t]):
                                    return f_id(f_id(x[:, t - 1]))
                            if down == np.sum(j) + 1:
                                return f_hat(x @ b[l - 1], u[k - 1])
                            if np.sum(j) + 2 <= down <= 2 ** s:
                                return np.ones(shape=[x.shape[0]])
                            raise AssertionError
                        raise AssertionError

                    if 0 <= np.sum(j) <= N:
                        B.append(f_down_up(X_train, 1, 0))
        B = np.array(B).T
        a = np.linalg.inv(B.T @ B + c_3) @ (B.T @ y_train)
        print(np.max(a))
        loss = np.mean(np.square(B @ a - y_train))
        print('loss:', loss)

        if loss < min_loss:
            min_loss = loss

    print('min loss:', min_loss)
