import numpy as np


def gen_sinusoidal(n, seed=None, sdev=0.2):
    """Generates sinusoidal data with random noise added to it.
    
    Parameters
    ----------
    n : int
        Number of data points to generate.

    seed : int
        Seed value for numpy random number generator.

    sdev : float
        Standard deviation Gaussian distribution.
    
    Returns
    -------
    x : Numpy array [n, 1]
        Feature vector.
    
    t : Numpy array [n, 1]
        Target vector.
    """
    if seed:
        np.random.seed(seed)
    x = np.linspace(0, 2 * np.pi, n).reshape([-1,1])
    t = np.random.normal(np.sin(x), sdev).reshape([-1,1])
    return x, t


def gen_sinusoidal_norm(n, seed=None, sdev=0.2, ):
    """ Generates sinusoidal data with random noise added to it.
        xi's are drawn from a uniform distribution.

    Parameters
    ----------
    n : int
        Number of data points to generate.

    seed : int
        Seed value for numpy random number generator.

    sdev : float
        Standard deviation Gaussian distribution.


    Returns
    -------
    x : Numpy array [n, 1]
        Feature vector.

    t : Numpy array [n, 1]
        Target vector.
    """
    if seed:
        np.random.seed(seed)
    x = np.random.uniform(0, 2 * np.pi, n).reshape([-1, 1])
    t = np.random.normal(np.sin(x), sdev).reshape([-1, 1])
    return x, t
