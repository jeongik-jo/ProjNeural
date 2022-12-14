import numpy as np
import os
from scipy.stats import iqr

train_data_size = 80
test_data_size = 20
valid_data_size = 10000

noise_strength = 0.05

is_m1 = True
is_m2 = False
is_m3 = False
is_m4 = False
is_m5 = False


if is_m1:
    def m(X):
        return np.log((0.2 * X[:, 0] + 0.9 * X[:, 1]) ** 2) + \
               np.cos(np.pi / np.log((0.5 * X[:, 0] + 0.3 * X[:, 1]) ** 2)) +\
               np.exp(1/50 * (0.7 * X[:, 0] + 0.7 * X[:, 1])) +\
               np.tan(np.pi * (0.1 * X[:, 0] + 0.3 * X[:, 1]) ** 4) / (0.1 * X[:, 0] + 0.3 * X[:, 1]) ** 2
    input_dim = 2
    noise_scale = 2.20
elif is_m2:
    def m(X):
        return np.tan(np.sin(np.pi * (0.2 * X[:, 0] + 0.5 * X[:, 1] - 0.6 * X[:, 2] + 0.2 * X[:, 3]))) + \
               (0.5 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3])) ** 3 + \
               1 / ((0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.3 * X[:, 2] + 0.25 * X[:, 3]) ** 2 + 4)
    input_dim = 4
    noise_scale = 1.96
elif is_m3:
    def m(X):
        return np.log(0.5 * (X[:, 0] + 0.3 * X[:, 1] + 0.6 * X[:, 2] + X[:, 3] - X[:, 4]) ** 2) + \
               np.sin(np.pi * (0.7 * X[:, 0] + X[:, 1] - 0.3 * X[:, 2] - 0.4 * X[:, 3] - 0.8 * X[:, 4])) +\
               np.cos(np.pi / (1 + np.sin(0.5 * (X[:, 1] + 0.9 * X[:, 2] - X[:, 4]))))
    input_dim = 5
    noise_scale = 2.85
elif is_m4:
    def m(X):
        return np.exp(0.2 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4] + X[:, 5])) +\
               np.sin(np.pi / 2 * (X[:, 0] - X[:, 1] - X[:, 2] + X[:, 3] - X[:, 4] - X[:, 5])) +\
               1 / ((0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.8 * X[:, 2] - 0.5 * X[:, 3] + 0.6 * X[:, 4] - 0.2 * X[:, 5]) ** 2 + 6) + \
               0.5 * (X[:, 0] + X[:, 2] - X[:, 4]) ** 3
    input_dim = 6
    noise_scale = 1.59
elif is_m5:
    def m(X):
        return X[:, 0] + 0.3
    input_dim = 1
    noise_scale = 0.99


def save_dataset():
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    epsilons = []
    for i in range(50):
        X_train = np.random.uniform(-1, 1, size=[train_data_size, input_dim])
        y_train = m(X_train) + noise_strength * noise_scale * np.random.normal(size=[train_data_size])

        X_test = np.random.uniform(-1, 1, size=[test_data_size, input_dim])
        y_test = m(X_test) + noise_strength * noise_scale * np.random.normal(size=[test_data_size])

        X_valid = np.random.uniform(-1, 1, size=[valid_data_size, input_dim])
        y_valid = m(X_valid)

        np.save('datasets/X_train_%d.npy' % i, X_train)
        np.save('datasets/y_train_%d.npy' % i, y_train)
        np.save('datasets/X_test_%d.npy' % i, X_test)
        np.save('datasets/y_test_%d.npy' % i, y_test)
        np.save('datasets/X_valid_%d.npy' % i, X_valid)
        np.save('datasets/y_valid_%d.npy' % i, y_valid)

        y_valid_noised = y_valid + noise_strength * noise_scale * np.random.normal(size=[valid_data_size])
        epsilons.append(np.mean(np.square(np.mean(y_train) - y_valid_noised)))
    np.savetxt('datasets/eps.txt', [np.median(epsilons)])


def load_dataset(i):
    X_train = np.load('datasets/X_train_%d.npy' % i)
    y_train = np.load('datasets/y_train_%d.npy' % i)
    X_test = np.load('datasets/X_test_%d.npy' % i)
    y_test = np.load('datasets/y_test_%d.npy' % i)
    X_valid = np.load('datasets/X_valid_%d.npy' % i)
    y_valid = np.load('datasets/y_valid_%d.npy' % i)

    return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)


def get_loss_scale():
    Xs = [np.random.uniform(-1, 1, size=[100, input_dim]) for _ in range(50)]
    print(np.median([iqr(m(X)) for X in Xs]))

if __name__ == "__main__":
    save_dataset()
    #get_loss_scale()
