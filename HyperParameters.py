import numpy as np

train_data_size = 100
test_data_size = 10000
noise_strength = 0.05

is_m1 = False
is_m2 = True
is_m3 = False
is_m4 = False

if is_m1:
    def m(X):
        return np.log((0.2 * X[:, 0] + 0.9 * X[:, 1]) ** 2) + \
               np.cos(np.pi / np.log((0.5 * X[:, 0] + 0.3 * X[:, 1]) ** 2)) +\
               np.exp(1/50 * (0.7 * X[:, 0] + 0.7 * X[:, 1])) +\
               np.tan(np.pi * (0.1 * X[:, 0] + 0.3 * X[:, 1]) ** 4) / (0.1 * X[:, 0] + 0.3 * X[:, 1]) ** 2
    input_dim = 2 # d
    noise_scale = 5.04
elif is_m2:
    def m(X):
        return np.tan(np.sin(np.pi * (0.2 * X[:, 0] + 0.5 * X[:, 1] - 0.6 * X[:, 2] + 0.2 * X[:, 3]))) + \
               (0.5 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3])) ** 3 + \
               1 / ((0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.3 * X[:, 2] + 0.25 * X[:, 3]) ** 2 + 4)
    input_dim = 4
    noise_scale = 5.57
elif is_m3:
    def m(X):
        return np.log(0.5 * (X[:, 0] + 0.3 * X[:, 1] + 0.6 * X[:, 2] + X[:, 3] - X[:, 4]) ** 2) + \
               np.sin(np.pi * (0.7 * X[:, 0] + X[:, 1] - 0.3 * X[:, 2] - 0.4 * X[:, 3] - 0.8 * X[:, 4])) +\
               np.cos(np.pi / (1 + np.sin(0.5 * (X[:, 1] + 0.9 * X[:, 2] - X[:, 4]))))
    input_dim = 5
    noise_scale = 6.8
elif is_m4:
    def m(X):
        return np.exp(0.2 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4] + X[:, 5])) +\
               np.sin(np.pi / 2 * (X[:, 0] - X[:, 1] - X[:, 2] + X[:, 3] - X[:, 4] - X[:, 5])) +\
               1 / ((0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.8 * X[:, 2] - 0.5 * X[:, 3] + 0.6 * X[:, 4] - 0.2 * X[:, 5]) ** 2 + 6) + \
               0.5 * (X[:, 0] + X[:, 2] - X[:, 4]) ** 3
    input_dim = 6
    noise_scale = 3.71


#--------------
knn_neighbor_sizes = [1, 2, 3, 4, 8, 12, 16, 20]
#--------------
fc_depth_size = 3
fc_unit_sizes = [3, 6, 9, 12, 15]
epoch = 100000