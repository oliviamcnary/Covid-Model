"""IHME model for COVID-19 deaths."""

import numpy as np
import numpy.testing as testing
from scipy.optimize import curve_fit
from scipy.special import erf


# COVID model function
def IHME(t, alpha, beta, p):
    """
    Calculate cumulative deaths with IHME model.

    Parameters
    ----------
    t: numpy array
        Number of days after start day
    alpha: float
        Time when rate of death increase is a maximum
    beta: float
        Growth parameter
    p: float
        Control for maximum death rate

    Returns
    -------
    c_death_est: numpy array
        Estimated cumulative number of deaths

    """
    assert isinstance(t, np.ndarray), 't must be a numpy array'
    assert isinstance(alpha, float), 'alpha must be a float'
    assert isinstance(beta, float), 'beta must be a float'
    assert isinstance(p, float), 'p must be a float'
    z = alpha*(t-beta)
    gaus = (2/np.sqrt(np.pi))*erf(z)
    c_death_est = (p/2)*(1 + gaus)
    return c_death_est


# minimizing function
def IHME_min(func, x, y, p0=None):
    """
    Calculate least-squares estimates for COVID model function.

    Parameters
    ----------
    func: function
        Gaussian error function
    x: numpy array
        independent data (days)
    y: numpy array
        dependent data (cumulative deaths)
    p0: list
        starting points for optimization

    Returns
    -------
    alpha: float
        Least-squares estimate for alpha
    beta: float
        Least-squares estimate for beta
    p: float
        Least-squares estimate for p
    """
    assert isinstance(x, np.ndarray), 'x must be a numpy array'
    assert isinstance(y, np.ndarray), 'y must be a numpy array'
    assert isinstance(p0, list) or p0 is None, 'p0 must be a list'
    popt, pcov = curve_fit(func, x, y, p0=p0)
    alpha, beta, p = popt
    return alpha, beta, p


def test_IHMEmin_alpha_output_value():
    """Test that IHME_min returns the correct alpha value."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    assert np.isclose(alpha, 0.046301063844839084, rtol=1e-07), '\
        Incorrect alpha output'


def test_IHMEmin_alpha_output_type():
    """Test that IHME_min returns a float for alpha."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    assert isinstance(alpha, float), 'Alpha output is not a float'


def test_IHMEmin_beta_output_value():
    """Test that IHME_min returns the correct beta."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    assert np.isclose(beta, 18.1930295594257, rtol=1e-07), '\
        Incorrect beta output'


def test_IHMEmin_beta_output_type():
    """Test that IHME_min returns a float for beta."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    assert isinstance(beta, float), 'Beta output is not a float'


def test_IHMEmin_p_output_value():
    """Test that IHME_min returns the correct p."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    assert np.isclose(p, 47.5019418620306, rtol=1e-07), '\
        Incorrect p output'


def test_IHMEmin_p_output_type():
    """Test that IHME_min returns a float for p."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    assert isinstance(p, float), 'p is not a float'


def test_IHMEmin_func_input_type():
    """Test that IHME_min doesn't allow non-function input for func."""
    func1 = 1
    func2 = 'a'
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    testing.assert_raises(TypeError, IHME_min, func1, x, y), '\
        Non-function input for func'
    testing.assert_raises(TypeError, IHME_min, func2, x, y), '\
        Non-function input for func'


def test_IHMEmin_x_input_type():
    """Test that IHME_min doesn't allow non-nparray input for x."""
    x1 = 2
    x2 = 'abc'
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    testing.assert_raises(AssertionError, IHME_min, IHME, x1, y), '\
        Non-numpy array input for x'
    testing.assert_raises(AssertionError, IHME_min, IHME, x2, y), '\
        Non-numpy array input for x'


def test_IHMEmin_y_input_type():
    """Test that IHME_min doesn't allow non-nparray input for y."""
    x = np.arange(10)
    y1 = 5
    y2 = 'wrong'
    testing.assert_raises(AssertionError, IHME_min, IHME, x, y1), '\
        Non-numpy array input for y'
    testing.assert_raises(AssertionError, IHME_min, IHME, x, y2), '\
        Non-numpy array input for y'


def test_IHMEmin_p0_input_type():
    """Test that IHME_min doesn't allow non-list input for p0."""
    x = np.arange(10)
    y = np.array([4, 4, 4, 6, 7, 7, 7, 8, 12, 12])
    p0 = 2
    testing.assert_raises(AssertionError, IHME_min, IHME, x, y, p0), '\
        p0 must be None or a list'


def test_IHME_c_death_est_output_value():
    """Test that IHME returns the correct c_death_est."""
    x = np.arange(5)
    y = np.array([4, 6, 7, 8, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    result = IHME(x, alpha, beta, p)
    exp = np.array([3.9, 5.5, 7.2, 9.1, 11.2])
    assert np.allclose(result, exp, rtol=0.1), 'Incorrect c_death_est'


def test_IHME_c_death_est_output_type():
    """Test that IHME returns a numpy array for c_death_est."""
    x = np.arange(5)
    y = np.array([4, 6, 7, 8, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    result = IHME(x, alpha, beta, p)
    assert isinstance(result, np.ndarray), 'c_death_est is not a numpy array'


def test_IHME_t_input_type():
    """Test that IHME doesn't allow non-nparray input for t."""
    x = np.arange(5)
    y = np.array([4, 6, 7, 8, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    t = [5, 10, 15, 20, 25]
    testing.assert_raises(AssertionError, IHME, t, alpha, beta, p), '\
        Non-numpy array input for t'


def test_IHME_alpha_input_type():
    """Test that IHME doesn't allow non-float input for alpha."""
    x = np.arange(5)
    y = np.array([4, 6, 7, 8, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    alpha1 = 'no'
    t = np.array([5, 10, 15, 20, 25])
    testing.assert_raises(AssertionError, IHME, t, alpha1, beta, p), '\
        Non-float input for alpha'


def test_IHME_beta_input_type():
    """Test that IHME doesn't allow non-float input for beta."""
    x = np.arange(5)
    y = np.array([4, 6, 7, 8, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    beta1 = 'incorrect'
    t = np.array([5, 10, 15, 20, 25])
    testing.assert_raises(AssertionError, IHME, t, alpha, beta1, p), '\
        Non-float input for beta'


def test_IHME_p_input_type():
    """Test that IHME doesn't allow non-float input for p."""
    x = np.arange(5)
    y = np.array([4, 6, 7, 8, 12])
    alpha, beta, p = IHME_min(IHME, x, y)
    p1 = 'heckno'
    t = np.array([5, 10, 15, 20, 25])
    testing.assert_raises(AssertionError, IHME, t, alpha, beta, p1), '\
        Non-float input for p'
