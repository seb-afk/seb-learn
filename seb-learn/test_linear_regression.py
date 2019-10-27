import numpy as np
from numpy.testing import assert_allclose
from linear_regression import LinearRegression
from data.data_generators import gen_sinusoidal
from data.data_processing import create_design_matrix


def test_fit():
    y = np.array([0.09459717, 0.50650243, 1.03329565, 0.52587828, 0.49264871, -0.64896441,
                  -0.86499999, -1.00885329, -0.80418399, 0.57436388]).reshape(-1, 1)
    Xmat = np.array([0., 0., 0.6981317, 0.48738787, 1.3962634, 1.94955149, 2.0943951,
                     4.38649084, 2.7925268, 7.79820595, 3.4906585, 12.18469679,
                     4.1887902, 17.54596338, 4.88692191, 23.88200571, 5.58505361,
                     31.19282379, 6.28318531, 39.4784176]).reshape(-1, 2)
    model = LinearRegression()
    model.fit(Xmat, y)
    fitted_weights = model.weights_
    correct_weights = np.array([[0.77483422], [-0.42288373], [0.03914334]])
    np.testing.assert_almost_equal(fitted_weights, correct_weights, decimal=8)
