#!/usr/bin/env python

import argparse

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, Variable


from net import MyNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    args = parser.parse_args()

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = MyNet()
    model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.SGD(lr=0.0001)
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Learning loop
    while train_iter.epoch < args.epoch:
        train_batch = train_iter.next()
        x_data = []
        t_data = []
        for data in train_batch:
            x_data.append(data[0].reshape((1, 28, 28)))
            if data[1] % 2 == 0:
                t_data.append([1])
            else:
                t_data.append([-1])

        x = Variable(cuda.to_gpu(np.array(x_data, dtype=np.float32)))
        t = Variable(cuda.to_gpu(np.array(t_data, dtype=np.float32)))

        y = model(x)

        model.cleargrads()
        loss = F.mean_squared_error(F.tanh(y), t)
        loss.backward()
        optimizer.update()

        if train_iter.is_new_epoch:
            print('epoch:{} train_loss:{} '.format(
                train_iter.epoch, loss.data), end='')

            sum_test_loss = 0
            sum_test_accuracy = 0
            test_itr = 0
            while True:
                test_batch = test_iter.next()
                x_test_data = []
                t_test_data = []
                t_test_acc_data = []
                for test_data in test_batch:
                    x_test_data.append(test_data[0].reshape((1, 28, 28)))
                    if test_data[1] % 2 == 0:
                        t_test_data.append([1])
                        t_test_acc_data.append([1])
                    else:
                        t_test_data.append([-1])
                        t_test_acc_data.append([0])

                x_test = Variable(cuda.to_gpu(np.array(x_test_data, dtype=np.float32)))
                t_test = Variable(cuda.to_gpu(np.array(t_test_data, dtype=np.float32)))
                t_test_acc = Variable(cuda.to_gpu(np.array(t_test_acc_data, dtype=np.int32)))

                y_test = model(x_test)
                sum_test_loss += F.mean_squared_error(F.tanh(y_test), t_test).data
                sum_test_accuracy += F.binary_accuracy(y_test, t_test_acc).data
                test_itr += 1

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{} val_accuracy:{}'.format(
                sum_test_loss / test_itr, sum_test_accuracy / test_itr))

if __name__ == '__main__':
    main()