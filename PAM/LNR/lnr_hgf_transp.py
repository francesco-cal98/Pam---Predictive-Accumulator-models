import numpy as np

def lnr_hgf_transp(r, ptrans):
    """transform parameters to native space"""
    # initialize nan array
    pvec = np.empty(len(ptrans))
    pvec[:] = np.nan
    pstruct = {}

    try:
        y = r['y'][:, 0]  # Extract the first column from the 'y' array
        y = np.delete(y, r['irr'])  # Remove indices specified in 'irr'
    except Exception as e:  # Catch any exception that occurs
        y = np.nan  # Set y to NaN if there is an error
    
    # Mantain "bv" in the same space
    pvec[0] = ptrans[0]
    pstruct['a'] = pvec[0]

    # Mantain "bi" in the same space
    pvec[1] = ptrans[1]
    pstruct['b_val'] = pvec[1]

    # Mantain "b1" in the same space
    pvec[2] = ptrans[2]
    pstruct['b'] = pvec[2]
    
    # Apply exponential transformation to "sigma"
    pvec[3] = np.exp(ptrans[3])
    pstruct['sigma'] = pvec[3]

    if not np.isnan(y.any()):
        pvec[4] = np.min(y)/(1+np.exp(-pvec[4]))
        pstruct['Ter'] = pvec[4]
    else:
        pvec[4] = ptrans[4]
        pstruct['Ter'] = pvec[4]
        
    return([pvec, pstruct])

