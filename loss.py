#! /usr/bin/env python3
import tensorflow as tf

pi_p = None


def puloss(y_true, y_pred):
    return positive_risk(y_true, y_pred) + negative_risk(y_true, y_pred)


def nnpuloss(y_true, y_pred):
    return (positive_risk(y_true, y_pred)
            + tf.nn.relu(negative_risk(y_true, y_pred)))


def pretrain_loss(y_true, y_pred):
    return tf.maximum(positive_risk(y_true, y_pred),
                      negative_risk(y_true, y_pred))


def error(y_true, y_pred):
    global pi_p

    n_positive = tf.maximum(1., tf.reduce_sum(y_true))
    n_unlabeled = tf.maximum(1., tf.reduce_sum(1 - y_true))
    y_positive = (1 - tf.sign(y_pred)) / 2
    y_unlabeled = (1 + tf.sign(y_pred)) / 2
    positive_risk = tf.reduce_sum(pi_p * y_true / n_positive * y_positive)
    negative_risk = tf.reduce_sum(
        ((1 - y_true) / n_unlabeled - pi_p * y_true / n_positive) * y_unlabeled)
    return positive_risk + negative_risk


def positive_risk(y_true, y_pred):
    global pi_p

    loss_func = tf.nn.sigmoid

    n_positive = tf.maximum(1., tf.reduce_sum(y_true))
    r_plus = loss_func(-y_pred)
    return tf.reduce_sum(pi_p * y_true / n_positive * r_plus)


def negative_risk(y_true, y_pred):
    global pi_p

    loss_func = tf.nn.sigmoid

    n_positive = tf.maximum(1., tf.reduce_sum(y_true))
    n_unlabeled = tf.maximum(1., tf.reduce_sum(1 - y_true))
    r_minus = loss_func(y_pred)
    return tf.reduce_sum(
        ((1 - y_true) / n_unlabeled - pi_p * y_true / n_positive) * r_minus)
