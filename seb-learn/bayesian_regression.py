import numpy as np


class BayesianRegression(object):
    """
    Attributes
    ----------
    TODO: Update this


    Returns
    -------
    self: object
    """

    def __init__(self):
        self.prior_mean_ = None
        self.prior_precision_ = None
        self.likelihood_precision_ = None
        self.posterior_mean_ = None
        self.posterior_covariance_ = None
        self.predictive_variance_ = None
        self.predictive_mean_ = None

    def fit(self, x_mat, y_vec, prior_precision, likelihood_precision):
        """
        Fits a bayesian linear regression model.

        Parameters
        ----------
        x_mat: d-dimensional array, shape = [number_samples, number_features]
            Feature vectors horizontally stacked.

        y_vec: 1d-array, shape = [n_samples]
            Target vector.

        prior_precision: scalar
            Precision parameter for prior distribution in w.

        likelihood_precision: scalar
            Precision parameter for the likelihood. TODO Check this.

        Returns
        -------
        self
        """
        xb_mat = np.hstack([np.full((x_mat.shape[0], 1), 1, dtype="float64"),
                           x_mat])
        self.prior_mean_ = 0
        self.prior_precision_ = prior_precision
        self.likelihood_precision_ = likelihood_precision
        self.posterior_covariance_ = (
            np.linalg.inv(prior_precision * np.identity(xb_mat.shape[1]) +
                          (likelihood_precision * xb_mat.T.dot(xb_mat))))
        self.posterior_mean_ = (
            np.ravel(likelihood_precision *
                     self.posterior_covariance_.dot(xb_mat.T).dot(y_vec)))
        return self

    def predict(self, x_mat, method="map"):
        """Predict target.

        Parameters
        ----------
        x_mat: Numpy array, shape = [n samples, n features]
            Feature vectors horizontally stacked.

        method : str, ["map", "random]
            "map": Use the predictive mean as weight vector.
            "random" Sample random weights from the predictive distribution.


        Returns
        -------
        predictive_mean : Numpy array, shape = [n samples]
            Predictions for the x_mat data.

        predictive_variance : Numpy array. shape = [n samples, n samples]
            Covariance matrix of predictive distribution.
        """
        xb_mat = np.hstack([np.full((x_mat.shape[0], 1), 1, dtype="float64"),
                           x_mat])
        self.predictive_variance_ = (
                1/self.likelihood_precision_ * np.identity(xb_mat.shape[0]) +
                xb_mat.dot(self.posterior_covariance_.dot(xb_mat.T)))
        if method == "map":
            self.predictive_mean_ = self.posterior_mean_.dot(xb_mat.T)
        elif method == "random":
            weight_vector = (
                np.random.multivariate_normal(self.posterior_mean_,
                                              self.posterior_covariance_))
            self.predictive_mean_ = weight_vector.dot(xb_mat.T)
        else:
            raise ValueError('Method not specified. Set to "map" or "random".')
        return self.predictive_mean_, self.predictive_variance_
