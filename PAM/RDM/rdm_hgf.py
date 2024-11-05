import numpy as np
from PAM.RDM.utl import utl_inverse_gaussian

def RDM_hgf(r, infStates, ptrans):
    """
    Calculates the log-probability of response speed y (in units of seconds)
    according to the Racing Diffusion Model.

    Parameters:
    r : dict
        Dictionary containing responses and irregular trials.
    infStates : np.ndarray
        Array that contains the parameters of the HGF.
    ptrans : np.ndarray
        Array that contains the parameters for the response model.
    
    Returns:
    logp : np.ndarray
        Log-probabilities of the model.
    yhat : np.ndarray
        Not used in this implementation.
    res : np.ndarray
        Not used in this implementation.
    """

    # Transform parameters to their native space
    a_a = np.exp(ptrans[0])
    b_a = ptrans[1]
    a_v = np.exp(ptrans[2])
    b_val = np.exp(ptrans[3])
    b_v = ptrans[4]

    # Initialize returned log-probabilities, predictions, and residuals as NaNs
    n = infStates.shape[0]
    logp = np.full(n, np.nan)
    yhat = np.full(n, np.nan)  # Not used
    res = np.full(n, np.nan)   # Not used

    # Weed irregular trials out from inferred states, responses, and inputs
    mu1hat = infStates[:, 0, 0]
    mu1hat[r['irr']] = np.nan

    # Extract response times and responses, excluding irregular trials
    rt = r['y'][:, 0]
    rt[r['irr']] = np.nan

    resp = r['y'][:, 1]
    resp[r['irr']] = np.nan

    # Fitting the non-decision time with the minimum value of estimated non-decision time
    Ter = np.min(rt) / (1 + np.exp(-ptrans[5]))
    rt = np.maximum(np.finfo(float).eps, rt - Ter)

    # Extract the trial list and remove irregular trials
    u = r['u'][:]
    u = u.astype(float)  # Convert the entire array to float
    u[r['irr']] = np.nan

    # Calculate trial-wise threshold for both the accumulators
    a_c1 = a_a + b_a * (.5 - mu1hat)
    a_c0 = a_a + b_a * (.5 - (1 - mu1hat)) 

    # Calculate drift rates for both accumulators
    drift_c1 = a_v + b_val*(u==1).astype(float) +  b_v * (mu1hat - .5)
    drift_c0 = a_v + b_val*(u==0).astype(float) +  b_v * ((1-mu1hat) - .5)

    # Calculate predicted log-likelihood
    logp_reg = np.full(len(u), np.nan)
    for ntrial in range(len(u)):
        if rt[ntrial] >= 0:
            # Extract the drifts for the specific trial
            drift_pdf = (resp[ntrial] == 1) * drift_c1[ntrial] + (resp[ntrial] == 0) * drift_c0[ntrial]
            drift_cdf = (resp[ntrial] == 1) * drift_c0[ntrial] + (resp[ntrial] == 0) * drift_c1[ntrial]

            # Extract the threshold for the specific trial
            a_pdf = (resp[ntrial] == 1) * a_c1[ntrial] + (resp[ntrial] == 0) * a_c0[ntrial]
            a_cdf = (resp[ntrial] == 1) * a_c0[ntrial] + (resp[ntrial] == 0) * a_c1[ntrial]

            # Compute the defective distribution using the inverse Gaussian function
            P = utl_inverse_gaussian.utl_inverse_gaussian_defective(rt[ntrial], drift_pdf, drift_cdf, a_pdf, a_cdf)

            # Correct if P < 0
            P = np.maximum(P, 0)

            if P > 0:
                logp_reg[ntrial] = np.log(P + np.finfo(float).eps)
            else:
                logp_reg[ntrial] = np.nan
        else:
            logp_reg[ntrial] = np.nan

    # Mark regular trials
    reg = ~np.isin(np.arange(n), r['irr'])
    logp[reg] = logp_reg

    return logp, yhat, res
