import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow.keras as kr
import Dataset
import numpy as np
import time

depths = [1]
learning_rates = [0.02]
unit_sizes = [5, 10, 25, 50, 75]
epoch = 100000
activation = tf.nn.sigmoid
reg_weight = 1e-6


def build_model(depth, units):
    model_output = model_input = kr.Input([Dataset.input_dim])
    for _ in range(depth):
        model_output = kr.layers.Dense(units=units, activation=activation,
                                       kernel_regularizer=kr.regularizers.L2(reg_weight),
                                       bias_regularizer=kr.regularizers.L2(reg_weight),
                                       )(model_output)
    model_output = tf.squeeze(kr.layers.Dense(units=1,
                                              kernel_regularizer=kr.regularizers.L2(reg_weight),
                                              bias_regularizer=kr.regularizers.L2(reg_weight),
                                              )(model_output))
    return kr.Model(model_input, model_output)


def train(X_train, y_train, X_test, y_test):
    min_loss = np.inf
    min_model = None
    min_depth = None
    min_units = None

    for depth in depths:
        for units in unit_sizes:
            for learning_rate in learning_rates:
                @tf.function
                def train_step(model, optimizer, X_train, y_train):
                    with tf.GradientTape() as tape:
                        y_pred = model(X_train)
                        loss = tf.reduce_mean(tf.square(y_pred - y_train))
                    optimizer.apply_gradients(
                        zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))

                model = build_model(depth, units)
                optimizer = kr.optimizers.SGD(learning_rate=learning_rate)
                for _ in range(epoch):
                    train_step(model, optimizer, X_train, y_train)

                test_loss = tf.reduce_mean(tf.square(model(X_test) - y_test))

                if test_loss < min_loss:
                    min_loss = test_loss
                    min_model = model
                    min_depth = depth
                    min_units = units
                    min_learning_rate = learning_rate

    print('test loss:\t', min_loss.numpy())
    print('depth:\t', min_depth)
    print('units:\t', min_units)
    print('learning rate:\t', min_learning_rate)

    return min_model


def validation(model, X_valid, y_valid):
    loss = tf.reduce_mean(tf.square(model(X_valid) - y_valid))
    print('valid loss:\t', loss.numpy())
    return loss


def main(i):
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = Dataset.load_dataset(i)

    X_train = tf.cast(X_train, 'float32')
    y_train = tf.cast(y_train, 'float32')
    X_test = tf.cast(X_test, 'float32')
    y_test = tf.cast(y_test, 'float32')
    X_valid = tf.cast(X_valid, 'float32')
    y_valid = tf.cast(y_valid, 'float32')

    start = time.time()
    model = train(X_train, y_train, X_test, y_test)
    print('train time:\t', time.time() - start)

    return validation(model, X_valid, y_valid)


if __name__ == "__main__":
    main(0)
