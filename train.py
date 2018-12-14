#! /usr/bin/env python3
"""Training script

Usage:
  train [--dataset=<dataset>] [--loss=<loss>] [--batch_size=<batch_size>]
        [--lr=<learning_rate>] [--pretrain=<pretrain>]

Options:
  --dataset=<dataset>        MNIST|epsilon|20News|CIFAR-10 [default: MNIST]
  --loss=<loss>              PN|uPU|nnPU [default: nnPU]
  --batch_size=<batch_size>  batch size [default: 30500]
  --lr=<learning_rate>       learning rate [default: 0.001]
  --pretrain=<pretrain>      pretrain|finetune|no [default: no]
  -h --help                  Show this screen.
"""
import docopt


def MNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train, y_test = y_train % 2 == 0, 1 - y_test % 2
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def Cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np.logical_or(y_train < 2, y_train > 7)
    y_test = np.logical_or(y_test < 2, y_test > 7)
    y_train, y_test = y_train.ravel(), y_test.ravel()
    return (x_train, y_train), (x_test, y_test)


def News20():
    with open('20news/train.data', 'r') as fin:
        x_train_sp = np.loadtxt(fin).astype(np.int)
    with open('20news/test.data', 'r') as fin:
        x_test_sp = np.loadtxt(fin).astype(np.int)
    with open('20news/train.label', 'r') as fin:
        y_train = np.loadtxt(fin).astype(
            np.uint8) <= 11  # ['alt.', 'comp.', 'misc.' and 'rec.']
    with open('20news/test.label', 'r') as fin:
        y_test = np.loadtxt(fin).astype(np.uint8) <= 11
    x_train = np.zeros((np.max(x_train_sp[:, 0]), 61188), dtype=np.uint8)
    x_test = np.zeros((np.max(x_test_sp[:, 0]), 61188), dtype=np.uint8)
    x_train[x_train_sp[:, 0] - 1, x_train_sp[:, 1] - 1] = x_train_sp[:, 2]
    x_test[x_test_sp[:, 0] - 1, x_test_sp[:, 1] - 1] = x_test_sp[:, 2]
    return (x_train, y_train), (x_test, y_test)


def Epsilon():
    x_train = np.fromfile(file='epsilon/traindata', dtype=np.float64).reshape(
        (-1, 2000))
    x_test = np.fromfile(file='epsilon/testdata', dtype=np.float64).reshape(
        (-1, 2000))
    y_train = np.fromfile(file='epsilon/trainlabel', dtype=np.int32)
    y_test = np.fromfile(file='epsilon/testlabel', dtype=np.int32)
    y_train, y_test = y_train > 0, y_test > 0
    return (x_train, y_train), (x_test, y_test)


def PNtrain(dataset: str, batch_size=30500, lr=1e-3):
    if dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test) = MNIST()
        model = MLP(n_layers=6, activation='relu', use_softmax=True)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    elif dataset == 'epsilon':
        (x_train, y_train), (x_test, y_test) = Epsilon()
        model = MLP(n_layers=6, activation='softsign', use_softmax=True)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    elif dataset == '20News':
        (x_train, y_train), (x_test, y_test) = News20()
        model = MLP(n_layers=5, activation='softsign', use_softmax=True)
        optimizer = tf.keras.optimizers.Adagrad(lr=lr)
    elif dataset == 'CIFAR-10':
        (x_train, y_train), (x_test, y_test) = Cifar10()
        model = CNN(use_softmax=True)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    else:
        raise ValueError('Incorrect argument!')

    pi_p = np.count_nonzero(y_train) / y_train.size
    pi_n = 1 - pi_p
    n_p = 1000
    n_n = int(np.round((pi_n / 2 / pi_p) ** 2 * n_p))
    p_index = np.random.choice(y_train.sum(), n_p)
    n_index = np.random.choice(np.logical_not(y_train).sum(), n_n)
    train_data_x = np.concatenate(
        (x_train[y_train][p_index], x_train[np.logical_not(y_train)][n_index]))
    train_data_y = np.concatenate((np.ones(n_p), np.zeros(n_n)))
    train_data_x, x_test = train_data_x / 255., x_test / 255.

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['acc'])
    if os.path.isdir('logs/' + dataset + '-PN'):
        print('Error: log dir exist')
        exit(1)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir="logs/{}/train".format(dataset + '-PN'))
    summary_writer = tf.summary.FileWriter(
        "logs/{}/test".format(dataset + '-PN'))

    for i in range(10000):
        model.fit(train_data_x, train_data_y, batch_size=batch_size,
                  epochs=i + 1, shuffle=True, callbacks=[tensorboard],
                  initial_epoch=i)
        lossval, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        summary = tf.Summary()
        summary.value.add(tag='test-loss', simple_value=lossval)
        summary.value.add(tag='test-accuracy', simple_value=acc)
        summary.value.add(tag='test-error', simple_value=1 - acc)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()


def PUtrain(dataset: str, func_loss, batch_size=30500, lr=1e-3, pretrain='no'):
    if dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test) = MNIST()
        model = MLP(n_layers=6, activation='relu', use_softmax=False)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    elif dataset == 'epsilon':
        (x_train, y_train), (x_test, y_test) = Epsilon()
        model = MLP(n_layers=6, activation='softsign', use_softmax=False)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    elif dataset == '20News':
        (x_train, y_train), (x_test, y_test) = News20()
        model = MLP(n_layers=5, activation='softsign', use_softmax=False)
        optimizer = tf.keras.optimizers.Adagrad(lr=lr)
    elif dataset == 'CIFAR-10':
        (x_train, y_train), (x_test, y_test) = Cifar10()
        model = CNN(use_softmax=False)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
    else:
        raise ValueError('Error: unknown dataset')
    if pretrain == 'pretrain':
        # Override optimizer
        optimizer = tf.keras.optimizers.Adagrad(lr=lr)
        lossstr = '-pretrain'
    elif func_loss is loss.puloss:
        lossstr = '-uPU'
    elif func_loss is loss.nnpuloss:
        lossstr = '-nnPU'
    elif func_loss is loss.positive_risk:
        lossstr = '-pRisk'
    else:
        raise ValueError('Error: unknown loss')
    foldername = dataset + lossstr
    if os.path.isdir('logs/' + foldername):
        print('Error: log dir exist')
        exit(2)
    # prior probability
    pi_p = np.count_nonzero(y_train) / y_train.size
    loss.pi_p = pi_p

    # choose first 1000 samples as training samples
    n_p, n_n = 1000, y_train.size

    # randomly choose n_p training samples as positive samples
    # the rest is unlabeled samples
    p_index = np.random.choice(y_train.sum(), n_p)
    train_data_x = np.concatenate((x_train[y_train][p_index], x_train))
    train_data_y = np.concatenate((np.ones(n_p), np.zeros(n_n)))

    # normalize the training data and the test data
    train_data_x, x_test = train_data_x / 255.0, x_test / 255.0

    model.compile(
        optimizer=optimizer,
        loss=func_loss,
        metrics=['acc', loss.positive_risk, loss.negative_risk, loss.error])

    # TensorBoard visualization
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir="logs/{}/train".format(foldername))
    summary_writer = tf.summary.FileWriter("logs/{}/test".format(foldername))
    callbacks = [tensorboard]

    # Pretrain related
    if pretrain == 'pretrain':
        saver = tf.keras.callbacks.ModelCheckpoint(
            'checkpoint/model.ckpt', monitor='loss', verbose=1,
            save_best_only=True, save_weights_only=True)
        callbacks.append(saver)
    elif pretrain == 'finetune':
        model.load_weights("checkpoint/model.ckpt")

    for i in range(10000):
        model.fit(train_data_x, train_data_y, batch_size=batch_size,
                  epochs=i + 1, shuffle=True, callbacks=callbacks,
                  initial_epoch=i)
        lossval, acc, prisk, nrisk, err = model.evaluate(x_test, y_test,
                                                         batch_size=batch_size)
        summary = tf.Summary()
        summary.value.add(tag='test-loss', simple_value=lossval)
        summary.value.add(tag='test-accuracy', simple_value=acc)
        summary.value.add(tag='test-positive-risk', simple_value=prisk)
        summary.value.add(tag='test-negative-risk', simple_value=nrisk)
        summary.value.add(tag='test-error', simple_value=err)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    import tensorflow as tf
    import os

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    from keras.datasets import mnist, cifar10
    import numpy as np

    from model import MLP, CNN
    import loss

    if args['--loss'] == 'PN':
        PNtrain(dataset=args['--dataset'], batch_size=int(args['--batch_size']),
                lr=float(args['--lr']))
    elif args['--pretrain'] == 'pretrain':
        PUtrain(dataset=args['--dataset'], func_loss=loss.pretrain_loss,
                batch_size=int(args['--batch_size']), lr=float(args['--lr']),
                pretrain='pretrain')
    elif args['--loss'] == 'uPU':
        PUtrain(dataset=args['--dataset'], func_loss=loss.puloss,
                batch_size=int(args['--batch_size']), lr=float(args['--lr']),
                pretrain=args['--pretrain'])
    elif args['--loss'] == 'nnPU':
        PUtrain(dataset=args['--dataset'], func_loss=loss.nnpuloss,
                batch_size=int(args['--batch_size']), lr=float(args['--lr']),
                pretrain=args['--pretrain'])
    elif args['--loss'] == 'prisk':
        PUtrain(dataset=args['--dataset'], func_loss=loss.positive_risk,
                batch_size=int(args['--batch_size']), lr=float(args['--lr']))
    else:
        raise ValueError()
