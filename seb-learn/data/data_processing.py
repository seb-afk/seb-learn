import numpy as np


def create_design_matrix(n_order, x):
    """Transforms a feature vector into a design matrix 
    (nth order polynomial).
    
    Parameters
    ----------
    n_order : int
        Order of the polynomial.
    x : 1-d Numpy array. [n_samples]
        Feature vector.

    Returns
    -------
    dmatrix : [n_samples, n_order]
    """
    dmatrix = np.empty(shape=(x.shape[0], n_order))
    for i in range(len(x)):
        for j in range(n_order):
            dmatrix[i, j] = x[i] ** (j+1)
    return dmatrix