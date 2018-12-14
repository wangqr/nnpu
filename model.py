#! /usr/bin/env python3
import tensorflow as tf


def MLP(n_layers: int, activation: str, use_softmax=False):
    layers = [tf.keras.layers.Flatten()]
    for i in range(n_layers - 1):
        layers.append(tf.keras.layers.Dense(300, activation=activation))
        # layers += [
        #     tf.keras.layers.Dense(300, activation=activation),
        #     tf.keras.layers.BatchNormalization()
        # ]
    if use_softmax:
        layers.append(tf.keras.layers.Dense(2, activation='softmax'))
    else:
        layers.append(tf.keras.layers.Dense(1, activation=None))
    return tf.keras.models.Sequential(layers)


def CNN(use_softmax=False):
    layers = [
        tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(96, 3, strides=2, padding='same',
                               activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, 3, strides=2, padding='same',
                               activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, 1, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(10, 1, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu')
    ]
    if use_softmax:
        layers.append(tf.keras.layers.Dense(2, activation='softmax'))
    else:
        layers.append(tf.keras.layers.Dense(1))
    return tf.keras.models.Sequential(layers)
