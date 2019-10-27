""" Perceptron implementaion
"""
import numpy as np


class Perceptron(object):
    """ Perceptron classifier. 

    Parameters
    ----------

    eta : float between 0.01 and 1.0
        Learning rate.

    n_iter : int
        Number of iterations before algorithm is stopped.

    seed_value : int
        Seed value for random weight initialisation.

    Attributes
    ----------

    weights_ : 1d-array
        weights_ after fitting the perceptron model to the data.

    errors_: list
        Number of misclassifications

    References
    ----------
    Mostly based on Python Machine Learning - Second Edition by 
    Sebastian Raschka and Vahid Mirjalili.
    """

    def __init__(self, eta=0.01, n_iter=50, seed_value=123):
        self.eta = eta
        self.n_iter = n_iter
        self.seed_value = seed_value
        # For later initialisation
        self.weights_ = None
        self.errors_ = None

    def fit(self, x_mat, y_vec):
        """Implementation of the perceptron learning algorithm.

        Parameters
        ----------

        x_mat : 2d-array, shape = [number_samples, number_features]
            Feature vectors stacked in a matrix.

        y_vec : 1d-array, shape = [n_samples]
            Target vector.

        Returns
        -------

        self : object
        """
        # Initialise weight vector with the first item reserved for the bias
        randomgenerator = np.random.RandomState(self.seed_value)
        self.weights_ = randomgenerator.normal(loc=0.0, scale=0.01,
                                              size=1 + x_mat.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0

            # Cycle through data and update weights_
            for x_i, target in zip(x_mat, y_vec):
                update = self.eta * (target - self.predict(x_i))
                self.weights_[1:] += update * x_i
                self.weights_[0] += update  # Update bias term
                errors += int(update != 0.0)

            self.errors_.append(errors)
        return None

    def predict(self, x_mat):
        """ Predicts new target values given a data matrix.

        Parameters
        ----------

        x_mat : 2d-array, shape = [number_samples, number_features]

        Returns
        -------


        """
        # W.T X + bias
        y_hat = np.dot(x_mat, self.weights_[1:]) + self.weights_[0]
        y_hat = np.where(y_hat >= 0.0, 1, -1)
        return y_hat
