import numpy as np

train_data_size = 80
test_data_size = 20
valid_data_size = 10000

noise_strength = 0.0

is_m1 = False
is_m2 = True
is_m3 = False
is_m4 = False

normalize_m = False


if is_m1:
    def _m(X):
        return np.log((0.2 * X[:, 0] + 0.9 * X[:, 1]) ** 2) + \
               np.cos(np.pi / np.log((0.5 * X[:, 0] + 0.3 * X[:, 1]) ** 2)) +\
               np.exp(1/50 * (0.7 * X[:, 0] + 0.7 * X[:, 1])) +\
               np.tan(np.pi * (0.1 * X[:, 0] + 0.3 * X[:, 1]) ** 4) / (0.1 * X[:, 0] + 0.3 * X[:, 1]) ** 2
    input_dim = 2
    noise_scale = 5.04
    m_std = 2.02
elif is_m2:
    def _m(X):
        return np.tan(np.sin(np.pi * (0.2 * X[:, 0] + 0.5 * X[:, 1] - 0.6 * X[:, 2] + 0.2 * X[:, 3]))) + \
               (0.5 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3])) ** 3 + \
               1 / ((0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.3 * X[:, 2] + 0.25 * X[:, 3]) ** 2 + 4)
    input_dim = 4
    noise_scale = 5.57
    m_std = 1.2348424304003454
elif is_m3:
    def _m(X):
        return np.log(0.5 * (X[:, 0] + 0.3 * X[:, 1] + 0.6 * X[:, 2] + X[:, 3] - X[:, 4]) ** 2) + \
               np.sin(np.pi * (0.7 * X[:, 0] + X[:, 1] - 0.3 * X[:, 2] - 0.4 * X[:, 3] - 0.8 * X[:, 4])) +\
               np.cos(np.pi / (1 + np.sin(0.5 * (X[:, 1] + 0.9 * X[:, 2] - X[:, 4]))))
    input_dim = 5
    noise_scale = 6.8
    m_std = 2.4229453384407367
elif is_m4:
    def _m(X):
        return np.exp(0.2 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4] + X[:, 5])) +\
               np.sin(np.pi / 2 * (X[:, 0] - X[:, 1] - X[:, 2] + X[:, 3] - X[:, 4] - X[:, 5])) +\
               1 / ((0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.8 * X[:, 2] - 0.5 * X[:, 3] + 0.6 * X[:, 4] - 0.2 * X[:, 5]) ** 2 + 6) + \
               0.5 * (X[:, 0] + X[:, 2] - X[:, 4]) ** 3
    input_dim = 6
    noise_scale = 3.71
    m_std = 1.8310473320070153


if normalize_m:
    def m(X):
        return _m(X) / m_std
else:
    def m(X):
        return _m(X)


def load_dataset():
    X_train = np.random.uniform(-1, 1, size=[train_data_size, input_dim])
    y_train = m(X_train) + noise_strength * noise_scale * np.random.normal(size=[train_data_size])

    X_test = np.random.uniform(-1, 1, size=[test_data_size, input_dim])
    y_test = m(X_test) + noise_strength * noise_scale * np.random.normal(size=[test_data_size])

    X_valid = np.random.uniform(-1, 1, size=[valid_data_size, input_dim])
    y_valid = m(X_valid)

    return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)
