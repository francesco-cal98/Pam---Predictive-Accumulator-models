import numpy as np

def RDM_hgf_transp(r, ptrans):
    """
    Transform parameters to their native space for the Drift Diffusion Model (DDM)

    The structure and methodologies are inspired from the HGF toolbox, open source code available as part of the TAPAS
    software collection: Frässle, S., et al. (2021). TAPAS: An Open-Source Software Package 
    for Translational Neuromodeling and Computational Psychiatry. Frontiers in Psychiatry, 12:680811. 
    https://www.translationalneuromodeling.org/tapas


    Parameters:
        - r (dict): contains the responses
        - ptrans (np.ndarray): Array that contains the parameters 
    """

    # Initialize empty arrays for transformed parameter vector and structure
    pvec = np.empty(len(ptrans))
    pvec[:] = np.nan
    pstruct = {}

    try:
        # Extract the first column of 'y' and remove irregular trials
        y = r['y'][:, 0]  
        y = np.delete(y, r['irr'])  # Remove irregular trials based on 'irr' indices
    except Exception as e:
        # Set y to NaN if there is an error
        y = np.nan

    # Apply exponential transformation to "a"
    pvec[0] = np.exp(ptrans[0])
    pstruct['a_a'] = pvec[0]

    # Transform "ba" in the range (-1, 1)
    pvec[1] = ptrans[1]
    pstruct['b_a'] = pvec[1]

    # Apply exponential transformation to "Vv"
    pvec[2] = np.exp(ptrans[2])
    pstruct['a_v'] = pvec[2]

    # Apply exponential transformation to "Vi"
    pvec[3] = np.exp(ptrans[3])
    pstruct['b_val'] = pvec[3]

    # Transform "bv" in the range (-1, 1)
    pvec[4] = ptrans[4]
    pstruct['b_v'] = pvec[4]

    # Transform "Ter" using a sigmoid, centered at the minimum value of y
    if not np.isnan(y).all():  # Check if 'y' is not all NaN
        pvec[5] = np.min(y) / (1 + np.exp(-ptrans[5]))
        pstruct['Ter'] = pvec[5]
    else:  # In case of NaN or missing values, pass 'Ter' directly
        pvec[5] = ptrans[5]
        pstruct['Ter'] = pvec[5]

    return ([pvec, pstruct])
