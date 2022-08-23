import numpy as np
import HyperParameters as hp


def load_dataset():
    X_train = np.random.uniform(-1, 1, size=[hp.train_data_size, hp.input_dim])
    y_train = hp.m(X_train) + hp.noise_strength * hp.noise_scale * np.random.normal(size=[hp.train_data_size])

    X_valid = np.random.uniform(-1, 1, size=[hp.valid_data_size, hp.input_dim])
    y_valid = hp.m(X_valid) + hp.noise_strength * hp.noise_scale * np.random.normal(size=[hp.valid_data_size])

    X_test = np.random.uniform(-1, 1, size=[hp.test_data_size, hp.input_dim])
    y_test = hp.m(X_test)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
