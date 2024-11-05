import numpy as np
from scipy.stats import lognorm

def utl_lnr_pdf(x, mu1, mu2, sigma):
    """
    Calculate the combined probabilities from two log-normal distributions.

    Parameters:
    x : float or array-like
        The point(s) at which to evaluate the probability density.
    mu1 : float
        The mean parameter of the first log-normal distribution.
    mu2 : float
        The mean parameter of the second log-normal distribution.
    sigma : float
        The standard deviation of the log-normal distributions.

    Returns:
    probs : float or array-like
        The calculated probabilities.
    """

    # Calculate the PDF of the first log-normal distribution
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu1))

    # Calculate the survival function of the second log-normal distribution
    survival = lognorm.sf(x, s=sigma, scale=np.exp(mu2))

    # Combine the results
    probs = pdf * survival

    return probs
