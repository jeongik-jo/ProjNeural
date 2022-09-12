import numpy as np

def m(X):
    return np.tan(np.sin(np.pi * (0.2 * X[:, 0] + 0.5 * X[:, 1] - 0.6 * X[:, 2] + 0.2 * X[:, 3]))) + \
           (0.5 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3])) ** 3 + \
           1 / ((0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.3 * X[:, 2] + 0.25 * X[:, 3]) ** 2 + 4)

input_dim = 4
noise_strength = 0.05
noise_scale = 5.57

epsilons = []
for _ in range(50):
    n = 10000
    X = np.random.uniform(-1, 1, size=[n, input_dim])
    y_true = m(X)
    y_noised = y_true + noise_strength * noise_scale * np.random.normal(size=[n])
    X_est = np.random.uniform(-1, 1, size=[n, input_dim])
    y_true_est = m(X_est)
    y_noised_est = y_true_est + noise_strength * noise_scale * np.random.normal(size=n)
    eps = np.mean((np.mean(y_noised_est) - y_noised) ** 2)
    epsilons.append(eps)

print(np.median(epsilons))
