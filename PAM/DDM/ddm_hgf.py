import numpy as np
from PAM.DDM.utl.utl_wfpt import utl_wfpt

from PAM.DDM.utl.scaled_sigmoid import scaled_sigmoid

def ddm_hgf(r, infStates, ptrans):
    """
    Calculates the log-probability of responses.
    
    Parameters:
    r (dict): Contains irregular trials ('irr'), responses ('y'), and trial information ('u').
    infStates (np.ndarray): Inferred states array.
    ptrans (np.ndarray): Parameter transformations.

    Returns:
    tuple: log-probabilities (logp), predictions (yhat), residuals (res).
    """
    # Transform parameters to their native space
    a_a = np.exp(ptrans[0])
    a_v = np.exp(ptrans[1])
    b_w = 2 / (1 + np.exp(-ptrans[2])) - 1
    b_a = ptrans[3]
    b_v = ptrans[4]

    # Initialize return values
    n = infStates.shape[0]
    logp = np.full(n, np.nan)
    yhat = np.full(n, np.nan)  # not used
    res = np.full(n, np.nan)   # not used

    # Weed out irregular trials from inferred states, responses, and inputs
    mu1hat = infStates[:, 0, 0]
    mu1hat[r['irr']] = np.nan
    
    rt = r['y'][:, 0]
    rt[r['irr']] = np.nan
    
    resp = r['y'][:, 1]
    resp[r['irr']] = np.nan
    
    # Fit the non-decision time with the minimum value of estimated non-decision time
    Ter = np.min(rt) / (1 + np.exp(-ptrans[5]))
    rt = np.maximum(np.finfo(float).eps, rt - Ter)

    # Extract trial list and remove irregular trials    
    u = r['u'][:]
    u = u.astype(float)  # Convert the entire array to float
    u[r['irr']] = np.nan


    # Calculate trial-wise starting point
    w = 0.5 + b_w * (mu1hat - 0.5)

    # Calculate trial-wise absorbing barrier
    precision = scaled_sigmoid(1.0 / (mu1hat * (1 - mu1hat)) - 4, 1) - 0.5
    a = a_a + b_a * precision

    # Calculate trial-wise drift
    v = u * (a_v + b_v * (mu1hat - 0.5)) - (1 - u) * (a_v + b_v * ((1 - mu1hat) - 0.5))

    # Calculate predicted log-likelihood
    logp_reg = np.full(len(u), np.nan)
    for ntrial in range(len(u)):
        if rt[ntrial] > 0:
            P = (utl_wfpt(rt[ntrial], -v[ntrial], a[ntrial], 1 - w[ntrial]) * resp[ntrial] + 
                 utl_wfpt(rt[ntrial], v[ntrial], a[ntrial], w[ntrial]) * (1 - resp[ntrial]))

            if P > 0:
                logp_reg[ntrial] = np.log(P + np.finfo(float).eps)
            else:
                logp_reg[ntrial] = np.nan

    # Update logp with logp_reg values where trials are not irregular
    reg = np.isin(np.arange(1, n + 1), r['irr'], invert=True)
    logp[reg] = logp_reg

    return logp, yhat, res
