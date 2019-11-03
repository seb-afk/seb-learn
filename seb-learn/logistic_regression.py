""" Logistic regression implementation
"""

import numpy as np


class LogisticRegression(object):
    """Logistic regression implementation.

    Attributes
    ----------

    weights_ : 1d-array
        Weights after fitting the linear regression model to the data.

    cost_ : list
    Sum-of-squares cost function value at each epoch
    Returns
    -------

    self: object
    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def activation(self, z):
        """Compute the sigmoid of z.

        Parameters
        ----------

        z: A scalar or numpy array of any size.

        Returns
        -------

        sigmoid(z)
        """

        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))


    def fit(self, x_mat, y_vec):
        """Fit logistic model to data.

        Parameters
        ----------

        x_mat: d-dimensional array, shape = [n_samples, n_features]
            Feature vectors stacked in a matrix.

        y_vec: 1d-array, shape = [n_samples]
            Target vector.

        Returns
        -------

        self: object
        """
        # Initialise variables
        n_samples, n_features = x_mat.shape
        self.cost_ = list()

        # Init weight vector
        rgen = np.random.RandomState(seed=self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.1, size=n_features + 1)

        # Repeat n_iter times
        for i in range(self.n_iter):

            # Calculate W.X + bias followed by the activation.
            net_input = np.dot(x_mat, self.weights_[1:]) + self.weights_[0]
            output = self.activation(net_input)

            # Calculate the gradients
            errors = y_vec - output
            dw = np.dot(x_mat.T, errors)
            dw0 = np.sum(errors)

            # Update the weights_
            self.weights_[1:] += self.eta * dw
            self.weights_[0] += self.eta * dw0

            # Calcultate the cost and append
            cost = (-y_vec.dot(np.log(output)) - ((1 - y_vec).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def predict(self, x_mat):
        """Predict class labels.

        Parameters
        ----------

        x_mat: d-dimensional array, shape = [n_samples, n_features]
            Feature vectors stacked in a matrix.

        Returns
        -------

        class_labels : n-dimensional array, shape = [1, n_samples]
            Class predictions.
        """
        net_input = np.dot(x_mat, self.weights_[1:]) + self.weights_[0]
        output = self.activation(net_input)
        class_labels = np.where(output >= 0.5, 1, 0)
        return class_labels
