import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import HyperParameters as hp
from itertools import product
import math

A = 1
N = 2
R = 10**6
r = 4
Ms = [2, 4, 6, 16]
I_n = 400
d = hp.input_dim
s = np.ceil(np.log2(N + 1)) # 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f_mult(x, y):
    return (R ** 2) / 4 * (1 + np.exp(-1.0)) ** 3 / (np.exp(-2.0) - np.exp(-1.0)) * \
    (sigmoid(2 * (x + y) / R + 1) - 2 * sigmoid((x + y) / R + 1)
     - sigmoid(2 * (x - y) / R + 1) + 2 * sigmoid((x - y) / R + 1))


def f_id(x):
    return 4 * R * sigmoid(x / R) - 2 * R


def f_relu(x):
    return f_mult(f_id(x), sigmoid(R * x))


for M in Ms:
    J = r * (M + 1) * math.comb(N + d, d)
    u = -np.sqrt(d) * A + np.arange(0, M + 1) * 2 * np.sqrt(d) * A / M

    def f_hat(x, y):
        return f_relu(M / (2 * np.sqrt(d) * A) * (x - y) + 1) - 2 * f_relu(M / (2 * np.sqrt(d) * A) * (x - y)) + \
               f_relu(M / (2 * np.sqrt(d) * A) * (x - y) - 1)

    for l in np.arange(1, r + 1):
        for j in product(np.arange(0, N + 1), repeat=d):

            def f_bottom_up(x, bottom, up):
                if up in np.arange(0, s):
                    if bottom in np.arange(1, 2 ** up + 1):
                        return f_mult(f_bottom_up(x, 2 * bottom - 1, up + 1), f_bottom_up(x, 2 * bottom, up + 1))
                if up == s:
                    for t in np.arange(1, d + 1):
                        if np.sum(j[:t-1]) + 1 <= bottom <= np.sum(j[:t]):
                            return f_id(f_id(x[t+1]))

                    if bottom == np.sum(j) + 1:
                        return f_hat(b_lt @ x, u[bottom - 1])
                    if bottom in np.arange(np.sum(j) + 2, 2 ** s + 1):
                        return 1

                raise AssertionError

            f_bottom_up(x, 1, 0)