from __future__ import print_function
import numpy as np
from layers.full import FullLayer
from layers.softmax import SoftMaxLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.relu import ReluLayer

class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        next_input = x
        for lay in self.layers:
            next_input = lay.forward(next_input)

        last_output = next_input


        if target is not None:
            last_output = self.loss.forward(last_output, target)

        return last_output

        #raise NotImplementedError

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        next_input = self.loss.backward()
        for i in range(len(self.layers)):
            #import pdb;pdb.set_trace()
            print(next_input)
            next_input = self.layers[len(self.layers)-i-1].backward(next_input)
            # self.layers[len(self.layers)-i-1].update_param(0.1)

        print(next_input)
        return next_input

        #raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        raise NotImplementedError

    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        raise NotImplementedError
