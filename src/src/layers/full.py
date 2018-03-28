import numpy as np

class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None
        self.n_i = n_i
        self.n_o = n_o

        # need to initialize self.W and self.b
        # self.W = np.zeros((n_o,n_i))
        self.W = np.random.uniform(0,np.sqrt(float(2)/(n_i+n_o)), [n_o, n_i]) # random no. between 0 & sqrt(2/(n_i+n_o))
        self.b = np.zeros((2,n_o))
       
    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = np.copy(x)
        wt = np.transpose(self.W)
        r = np.empty(shape=(x.shape[0],self.n_o))
        
        for i in range(x.shape[0]):
            # print row_x
            a = np.dot(x[i], wt)
            res = a + self.b[i]
            r[i] = res

        # r += self.b
        return r

    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        res = np.zeros(y_grad.shape)
        for i in range(self.x.shape[0]):
            res += y_grad

        self.b_grad = res
        self.W_grad = np.zeros(self.W.shape)

        yt = np.transpose(y_grad)
        row_x = np.zeros((1, self.x.shape[1]))
        row_y = np.zeros((1, y_grad.shape[1]))
        res = np.zeros(self.W.shape)
        for i in range(self.x.shape[0]):
            row_x[0] =  self.x[i]
            row_y[0] = y_grad[i]
            yt = np.transpose(row_y)
            #  = np.transpose(y_grad[i])

            res += np.dot(yt, row_x)


        self.W_grad = res
        return np.dot(y_grad, self.W) 

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.b = self.b - lr*(self.b_grad)
        self.W = self.W - lr*(self.W_grad)