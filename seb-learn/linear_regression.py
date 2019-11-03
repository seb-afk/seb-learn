import numpy as np


class LinearRegression(object):
    """ Ordinary least square linear regression implementation.

    Attributes
    ----------
    weights_ : 1d-array
        Weights after fitting the linear regression model to the data.

    Returns
    -------
    self: object

    """

    def __init__(self):
        self.weights_ = None

    def fit(self, x_mat, y_vec):
        """Analytic solution for linear regression.

        Parameters
        ----------
        x_mat: Numpy array, shape = [n samples, n features]
            Feature vectors horizontally stacked.

        y_vec: Numpy array, shape = [n_samples]
            Target vector.

        Returns
        -------
        self: object
        """
        # Add x0 = 1 to compute bias term w0
        xb_mat = np.hstack([np.full((x_mat.shape[0], 1), 1, dtype="float64"),
                           x_mat])
        self.weights_ = np.dot(np.linalg.pinv(xb_mat), y_vec)
        return self

    def predict(self, x_mat):
        """Predict target.

        Parameters
        ----------
        x_mat: d-dimensional array, shape = [number_samples, number_features]
            Feature vectors stacked in a matrix.

        Returns
        -------
        1d-array, shape = [number_samples]
            Predictions for the x_mat data.

        """
        return np.dot(x_mat, self.weights_[1:]) + self.weights_[0]
