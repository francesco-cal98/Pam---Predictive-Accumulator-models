import numpy as np
from scipy.stats import norm

def utl_inverse_gaussian_defective(x, drift_pdf, drift_cdf, threshold1, threshold2):
    """
    Python equivalent of the MATLAB function utl_inverse_gaussian_defective.

    Args:
    x : array-like, values at which to compute probabilities (in seconds).
    drift_pdf : array-like, drift rates for the PDF component.
    drift_cdf : array-like, drift rates for the CDF component.
    threshold1 : float, the first threshold parameter.
    threshold2 : float, the second threshold parameter.

    Returns:
    probs : array-like, computed probability values.
    """
    # PDF component
    pdf = (threshold1) / np.sqrt(2 * np.pi * (x**3)) * np.exp(-0.5 * ((drift_pdf * x - threshold1)**2) / x)

    # CDF component
    cdf_part1 = norm.cdf((drift_cdf * x - threshold2) / np.sqrt(x))
    cdf_part2 = np.exp(2 * drift_cdf * threshold2) * norm.cdf((-drift_cdf * x - threshold2) / np.sqrt(x))
    cdf = cdf_part1 + cdf_part2

    # Calculate the final probabilities
    probs = pdf * (1 - cdf)
    
    return probs
