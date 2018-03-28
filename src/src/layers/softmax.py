import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """

        y = np.copy(x)
        ymax = np.amax(y)
        ydash = y - ymax
        ydash_exp = np.exp(ydash)

        r = np.zeros(y.shape)
        i = 0
        j = 0
        for i in range(y.shape[0]):
            exp_sum = np.sum((ydash_exp[i]))
            for j in range(y.shape[1]):
                r[i][j] = ydash_exp[i][j]/exp_sum
        self.y = r
        return r

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        jacobian_diag = np.zeros(shape=(self.y.shape[1],self.y.shape[1]))

        jacobian_main = np.zeros(self.y.shape)


        for k in range(self.y.shape[0]):

            for i in range(self.y.shape[1]):
                for j in range(self.y.shape[1]):
                    if i == j:
                        jacobian_diag[i][j] = self.y[k][i]
                    else: 
                        jacobian_diag[i][j] = 0

            jacobian = jacobian_diag - np.dot(np.transpose([self.y[k]]), [self.y[k]])
            dot = np.dot(y_grad,jacobian)
            jacobian_main[k] = dot[k]

        return jacobian_main

    def update_param(self, lr):
        pass  # no learning for softmax layer
