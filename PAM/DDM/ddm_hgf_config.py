import numpy as np

def ddm_hgf_config():
    # Config structure as a dictionary
    c = {}
    
    # Model's name
    c['model'] = 'ddm: hgf'

    # Intercept of "a"
    c['a_amu'] = 0
    c['a_asa'] = 4

    # Slope of "a"
    c['a_vmu'] = 0
    c['a_vsa'] = 4

    # Drift V for valid response (resp == input)
    c['b_wmu'] = 0
    c['b_wsa'] = 4

    # Drift V for invalid response (resp != input)
    c['b_amu'] = 0
    c['b_asa'] = 4

    # Influence of muhat on "v"
    c['b_vmu'] = 0
    c['b_vsa'] = 4

    # Non-decision time
    c['Ter_mu'] = 0
    c['Ter_sa'] = 4

    ######## SET PRIORS #########
    c['priormus'] = np.concatenate([
        np.array([c['a_amu']]), 
        np.array([c['a_vmu']]), 
        np.array([c['b_wmu']]), 
        np.array([c['b_amu']]), 
        np.array([c['b_vmu']]), 
        np.array([c['Ter_mu']])
    ], axis=0)

    c['priorsas'] = np.concatenate([
        np.array([c['a_asa']]), 
        np.array([c['a_vsa']]), 
        np.array([c['b_wsa']]), 
        np.array([c['b_asa']]), 
        np.array([c['b_vsa']]), 
        np.array([c['Ter_sa']])
    ], axis=0)

    # Model file handle (function name as string)
    c['obs_fun'] = 'ddm_hgf'

    # Handle to function that transforms observation parameters to their native space
    c['transp_obs_fun'] = 'ddm_hgf_transp'

    return(c)

