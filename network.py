#!/usr/bin/env python

"""
Deep neural network for the classification
of the kaggle dataset: cats and dogs

Author: Toni Gabas.  a.gabas@aist.go.jp
"""

from __future__ import print_function
from tqdm import tqdm

import time
import numpy as np
import theano
import theano.tensor as T
import lasagne


"""
Helper function to get minibatches over the full batch.
"""


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if batchsize > len(inputs):
        batchsize = len(inputs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def load_data(inputsPath, labelsPath):
        inputs = np.load(inputsPath)
        inputs = np.asarray(inputs, dtype=theano.config.floatX)/256
        labels = np.load(labelsPath)
        N, chans, sizeX, sizeY = inputs.shape
        split_indices = (int(N*0.8), int((N*0.8)+(N*0.1)))
        inputs = inputs.reshape((-1, 3, sizeX, sizeY))
        labels = labels.astype(np.int32)
        X_train, X_val, X_test = np.split(inputs, split_indices)
        y_train, y_val, y_test = np.split(labels, split_indices)
        return X_train, y_train, X_val, y_val, X_test, y_test


class catVSdogCNN(object):
        def __init__(self, sizeX, sizeY):
            print("Building model and compiling functions...")
            self.sizeX = sizeX  # Patch size in X direction
            self.sizeY = sizeY  # Patch size in Y direction
            self.__input_var = T.tensor4('inputs')
            self.__target_var = T.ivector('targets')
            self.__network = self.__build_cnn()

            # Create structure for training parameters
            # with default values.
            self.trainParams = {
                'learning_rate': 0.0001,
                'momentum': 0.8,
                'batch_size': 50,
                'num_epochs': 400
            }

            # Create a loss expression for training, i.e., a scalar objective
            # we want to minimize (for our multi-class problem, it is
            # the cross-entropy loss):
            self.__prediction = lasagne.layers.get_output(self.__network)
            self.__loss = lasagne.objectives.binary_crossentropy(
                                        T.clip(self.__prediction, 1e-7, 1-(1e-7)),
                                        self.__target_var)
            self.__loss = self.__loss.mean()
            # Could add some weight decay as well here,
            # see lasagne.regularization.

            # Create update expressions for training, i.e., how to modify the
            # parameters at each training step. Here, using Stochastic Gradient
            # Descent (SGD) with Nesterov momentum,
            self.__params = lasagne.layers.get_all_params(
                                                                self.__network,
                                                                trainable=True)
            self.__updates = lasagne.updates.rmsprop(
                            self.__loss,
                            self.__params,
                            learning_rate=self.trainParams["learning_rate"])
                            #momentum=self.trainParams["momentum"])

            # Create a loss expression for validation/testing. The difference
            # here is that we do a deterministic forward pass through the
            # network, disabling dropout layers.
            self.__test_prediction = lasagne.layers.get_output(
                                            self.__network,
                                            deterministic=True)
            self.__test_loss = lasagne.objectives.binary_crossentropy(
                                            T.clip(self.__test_prediction, 1e-7, 1-(1e-7)),
                                            self.__target_var)
            self.__test_loss = self.__test_loss.mean()
            # Also create an expression for the classification accuracy:
            self.__test_acc = lasagne.objectives.binary_accuracy(self.__test_prediction,self.__target_var)
            self.__test_acc = self.__test_acc.mean()
            #self.__test_acc = T.mean(T.eq(
            #                                    T.argmax(
            #                                          self.__test_prediction,
            #                                          axis=1),
            #                                    self.__target_var),
            #                         dtype=theano.config.floatX)

            # Compile a function performing a training step on a mini-batch
            # (by giving the updates dictionary) and returning
            # the corresponding training loss:
            self.__train_fn = theano.function(
                            [self.__input_var, self.__target_var],
                            self.__loss,
                            updates=self.__updates)
                            #allow_input_downcast=True)

            # Compile a second function computing the validation loss
            # and accuracy:
            self.__val_fn = theano.function(
                [self.__input_var, self.__target_var],
                [self.__test_loss, self.__test_acc])
                #allow_input_downcast=True)

            # Compile one last function returning the prediction for
            # testing without labels available.
            self.__pred_fn = theano.function(
                [self.__input_var],
                self.__test_prediction)
                #allow_input_downcast=True)


        def __build_cnn(self):
            # create a CNN of 3 convolution + pooling stages
            # and a fully-connected hidden layer in front of the output layer.
            # Input layer:
            network = lasagne.layers.InputLayer(
                                                shape=(
                                                       None,
                                                       3,
                                                       self.sizeX,
                                                       self.sizeY),
                                                input_var=self.__input_var)

            # Convolutional layer with 32 kernels of size 3x3.
            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=32,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify,
                                 W=lasagne.init.GlorotUniform())

            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=32,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify,
                                 W=lasagne.init.GlorotUniform())

            # Batch normalization layer
            #network = lasagne.layers.BatchNormLayer(network)
            # Max-pooling layer of factor 2 in both dimensions:
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # Another convolution with 64 3x3 kernels, and another 2x2 pooling:
            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=64,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify)

            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=64,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify)

            # Batch normalization layer
            #network = lasagne.layers.BatchNormLayer(network)
            # Max-pooling layer of factor 2 in both dimensions:
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # Convolution with 128 3x3 kernels, and another 2x2 pooling:
            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=128,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify)

            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=128,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify)

            # Batch normalization layer
            #network = lasagne.layers.BatchNormLayer(network)
            
            # Max-pooling layer of factor 2 in both dimensions:
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # Convolution with 256 3x3 kernels, and another 2x2 pooling:
            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=128,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify)

            network = lasagne.layers.Conv2DLayer(
                                 network,
                                 num_filters=128,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.rectify)

            # Batch normalization layer
            #network = lasagne.layers.BatchNormLayer(network)
            
            # Max-pooling layer of factor 2 in both dimensions:
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


            # Fully-connected layer of 256 units and 50% dropout on its inputs:
            network = lasagne.layers.DenseLayer(
                                 lasagne.layers.dropout(network, p=0.5),
                                 num_units=256,
                                 nonlinearity=lasagne.nonlinearities.rectify)

            network = lasagne.layers.DenseLayer(
                                 lasagne.layers.dropout(network, p=0.5),
                                 num_units=256,
                                 nonlinearity=lasagne.nonlinearities.rectify)

            # Batch normalization layer
            #network = lasagne.layers.BatchNormLayer(network)
            # The output layer with 50% dropout on its inputs:
            network = lasagne.layers.DenseLayer(
                                 lasagne.layers.dropout(network, p=0),
                                 num_units=1,
                                 nonlinearity=lasagne.nonlinearities.sigmoid)
            return network

        def train(self, inputsPath, labelsPath=None):
            print("Loading data...")
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(
                                                                inputsPath,
                                                                labelsPath)

            # Finally, launch the training loop.
            print("Starting training...")
            # We iterate over epochs:
            num_epochs = self.trainParams["num_epochs"]
            batchsize = self.trainParams["batch_size"]
            len_epoch = (len(X_train) - batchsize + 1) / batchsize
            for epoch in range(num_epochs):
                # In each epoch, we do a full pass over the training data:
                train_err = 0
                train_batches = 0
                start_time = time.time()
                for batch in tqdm(iterate_minibatches(X_train,
                                                      y_train,
                                                      batchsize,
                                                      shuffle=False),
                                  total=len_epoch):
                    inputs, targets = batch
                    #print(targets[0:15])
                    #print(self.__pred_fn(inputs)[0:15])
                    train_err += self.__train_fn(inputs, targets)
                    train_batches += 1

                # And a full pass over the validation data:
                val_err = 0
                val_acc = 0
                val_batches = 0
                for batch in iterate_minibatches(
                                                            X_val,
                                                            y_val,
                                                            batchsize,
                                                            shuffle=False):
                    inputs, targets = batch
                    err, acc = self.__val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(
                                                train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(
                                                val_err / val_batches))
                print("  validation accuracy:\t\t{:.6f} %".format(
                                                val_acc / val_batches * 100))



                # Save the model after a number of epochs
                if epoch % 50 == 0:
                        params = lasagne.layers.get_all_param_values(
                            self.__network)
                        modelname = "model_" + str(epoch) + "_epoch"
                        np.save(modelname, params)

            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(
                                                    X_test,
                                                    y_test,
                                                    batchsize,
                                                    shuffle=False):
                inputs, targets = batch
                err, acc = self.__val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
            params = lasagne.layers.get_all_param_values(self.__network)
            np.save("model", params)

        def test(self, inputs):
            # TODO warining empty model
            # lasagne.layers.set_all_param_values(network,model)

            return self.__pred_fn(inputs)

if __name__ == '__main__':
    net = catVSdogCNN(64,64)
    net.train("input/ctraindata.npy","input/clabels.npy")
