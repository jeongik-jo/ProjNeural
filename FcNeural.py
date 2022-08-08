import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import Dataset
import numpy as np
from scipy.stats import iqr

"""
class EqDense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True):
        super(EqDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.units]), name=self.name + '_w')
        self.he_std = tf.sqrt(1.0 / tf.cast(input_shape[-1], 'float32'))

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_vector = tf.matmul(inputs, self.w) * self.he_std
        if self.use_bias:
            feature_vector = feature_vector + self.b

        return self.activation(feature_vector)


def build_model(units, depth):
    model_output = model_input = kr.Input([hp.input_dim])
    for _ in range(depth):
        model_output = EqDense(units=units, activation=tf.nn.leaky_relu)(model_output)
    model_output = tf.squeeze(EqDense(units=1)(model_output))
    return kr.Model(model_input, model_output)
"""

def build_model(units, depth):
    model_output = model_input = kr.Input([hp.input_dim])
    for _ in range(depth):
        model_output = kr.layers.Dense(units=units, activation=tf.nn.leaky_relu)(model_output)
    model_output = tf.squeeze(kr.layers.Dense(units=1)(model_output))
    return kr.Model(model_input, model_output)


def evaluate(X_train, y_train, X_test, y_test):
    X_train = tf.cast(X_train, 'float32')
    y_train = tf.cast(y_train, 'float32')
    X_test = tf.cast(X_test, 'float32')
    y_test = tf.cast(y_test, 'float32')

    unit_sizes = []
    mean_l2_errors = []
    median_l2_errors = []
    l2_iqrs = []

    for unit_size in hp.fc_unit_sizes:
        model = build_model(units=unit_size, depth=hp.fc_depth_size)
        optimizer = kr.optimizers.SGD(learning_rate=1e-3)

        @tf.function
        def train(model, optimizer, X_train, y_train):
            with tf.GradientTape() as tape:
                y_pred = model(X_train)
                loss = tf.reduce_mean(tf.square(y_pred - y_train))
            optimizer.apply_gradients(
                zip(tape.gradient(loss, model.trainable_variables),
                    model.trainable_variables)
            )

        for _ in range(hp.epoch):
            train(model, optimizer, X_train, y_train)
        l2_errors = tf.square(model(X_test) - y_test)

        unit_sizes.append(unit_size)
        mean_l2_errors.append(np.mean(l2_errors))
        median_l2_errors.append(np.median(l2_errors))
        l2_iqrs.append(iqr(l2_errors))

    i = np.argmin(median_l2_errors)
    print('unit size :', unit_sizes[i])
    print('median l2 :', median_l2_errors[i])
    print('mean l2 :', np.array(mean_l2_errors[i]))
    print('iqr :', l2_iqrs[i])


def main():
    (X_train, y_train), (X_test, y_test) = Dataset.load_dataset()
    evaluate(X_train, y_train, X_test, y_test)
main()