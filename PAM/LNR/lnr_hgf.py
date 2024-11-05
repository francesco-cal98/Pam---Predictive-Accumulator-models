from PAM.LNR.utl import utl_lnr_pdf
import numpy as np

def lnr_hgf(r, infStates, ptrans):
    """
    Calculates the log-probability of responses.
    
    Parameters:
    r (dict): Contains irregular trials ('irr'), responses ('y'), and trial information ('u').
    infStates (np.ndarray): Inferred states array.
    ptrans (np.ndarray): Parameter transformations.

    Returns:
    tuple: log-probabilities (logp), predictions (yhat), residuals (res).

    The structure and methodologies are inspired from the HGF toolbox, open source code available as part of the TAPAS
    software collection: Frässle, S., et al. (2021). TAPAS: An Open-Source Software Package 
    for Translational Neuromodeling and Computational Psychiatry. Frontiers in Psychiatry, 12:680811. 
    https://www.translationalneuromodeling.org/tapas
<
    """
    
    # Transform parameters to their native space
    a = ptrans[0]
    b_val = ptrans[1]
    b = ptrans[2]
    sigma = np.exp(ptrans[3])

    # Initialize returned log-probabilities, predictions, and residuals as NaNs
    n = infStates.shape[0]
    logp = np.full(n, np.nan)
    yhat = np.full(n, np.nan)  # not used
    res = np.full(n, np.nan)    # not used

    # Weed irregular trials out from inferred states, responses, and inputs
    mu1hat = infStates[:, 0, 0]
    mu1hat[r['irr']] = np.nan

    rt = r['y'][:, 0]
    rt[r['irr']] = np.nan

    resp = r['y'][:, 1]
    resp[r['irr']] = np.nan

    # Fitting the non-decision time with the minimum value of estimated non-decision time
    Ter = np.min(rt) / (1 + np.exp(-ptrans[4]))
    rt = np.maximum(np.finfo(float).eps, rt - Ter)

    # Extract the trial list and remove the irregular trials
    u = r['u'][:]
    u = u.astype(float)  # Convert the entire array to float
    u[r['irr']] = np.nan

    # Calculate trial-wise drift rates for the two accumulators
    mu_c1 = a + np.multiply(b_val,(u==1).astype(float)) + np.multiply(b,(.5-mu1hat))
    mu_c0 = a + np.multiply(b_val,(u==0).astype(float)) + np.multiply(b,(.5-(1-mu1hat)))


    # Calculate predicted log-likelihood
    logp_reg = np.full(len(u), np.nan)
    for ntrial in range(len(u)):
        if rt[ntrial] > 0:
            mu_pdf = (resp[ntrial] == 1) * mu_c1[ntrial] + (resp[ntrial] == 0) * mu_c0[ntrial]
            mu_cdf = (resp[ntrial] == 1) * mu_c0[ntrial] + (resp[ntrial] == 0) * mu_c1[ntrial]
            P = utl_lnr_pdf.utl_lnr_pdf(rt[ntrial], mu_pdf, mu_cdf, sigma)

            # Correct if P < 0
            P = np.maximum(P, 0)

            # Linear deterministic time distribution
            if P > 0:
                logp_reg[ntrial] = np.log(P + np.finfo(float).eps)
            else:
                logp_reg[ntrial] = np.nan
        else:
            logp_reg[ntrial] = np.nan

    # Mark regular trials
    reg = ~np.isin(np.arange(1, n + 1), r['irr'])
    logp[reg] = logp_reg[reg]

    return logp, yhat, res