import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow.keras as kr
import Dataset
import numpy as np
from scipy.stats import iqr


depth = 3
learning_rate = 1e-3
unit_sizes = [3, 6, 9, 12, 15]
epoch = 100000


def build_model(units):
    model_output = model_input = kr.Input([Dataset.input_dim])
    for _ in range(depth):
        model_output = kr.layers.Dense(units=units, activation=tf.nn.leaky_relu)(model_output)
    model_output = tf.squeeze(kr.layers.Dense(units=1)(model_output))
    return kr.Model(model_input, model_output)


def train(X_train, y_train, X_valid, y_valid):
    X_train = tf.cast(X_train, 'float32')
    y_train = tf.cast(y_train, 'float32')
    X_valid = tf.cast(X_valid, 'float32')
    y_valid = tf.cast(y_valid, 'float32')

    min_loss = np.inf
    min_model = None
    min_units = None

    for units in unit_sizes:
        @tf.function
        def train_step(model, optimizer, X_train, y_train):
            with tf.GradientTape() as tape:
                y_pred = model(X_train)
                loss = tf.reduce_mean(tf.square(y_pred - y_train))
            optimizer.apply_gradients(
                zip(tape.gradient(loss, model.trainable_variables),
                    model.trainable_variables))

        model = build_model(units)
        optimizer = kr.optimizers.SGD(learning_rate=learning_rate)
        for _ in range(epoch):
            train_step(model, optimizer, X_train, y_train)

        loss = tf.reduce_mean(tf.square(model(X_valid) - y_valid))

        if loss < min_loss:
            min_loss = loss
            min_model = model
            min_units = units
    print('valid loss:', min_loss.numpy())
    print('units:', min_units)

    return min_model


def test(model, X_test, y_test):
    X_test = tf.cast(X_test, 'float32')
    y_test = tf.cast(y_test, 'float32')

    losses = tf.square(model(X_test) - y_test)
    print('\nmse:\t', np.mean(losses))
    print('median:\t', np.median(losses))
    print("iqr:\t", iqr(losses))


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = Dataset.load_dataset()
    model = train(X_train, y_train, X_valid, y_valid)
    test(model, X_test, y_test)


main()
