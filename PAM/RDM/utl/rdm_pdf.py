import numpy as np

def rdm_pdf(x, drift_pdf, threshold):
    """
    Compute the probability density function (PDF) of the Racing Diffusion Model.

    Parameters:
    x : np.ndarray
        Array of time values.
    drift_pdf : float
        Drift rate for the accumulator.
    threshold : float
        Threshold value for the accumulator.

    Returns:
    pdf : np.ndarray
        Array of PDF values for each time value in x.
    """
    
    pdf = (threshold) / np.sqrt(2 * np.pi * (x ** 3)) * np.exp(-0.5 * ((drift_pdf * x - threshold) ** 2) / x)
    
    return pdf
