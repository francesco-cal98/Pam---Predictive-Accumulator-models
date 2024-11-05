import numpy as np

def rdm_hgf_config():

    '''
    Contains the configuration for the Racing Diffusion Model according to:
    Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling models without random between-trial variability: the racing diffusion model of speeded decision making. 
    Psychonomic Bulletin & Review, 27(5), 911–936
    
    This configuration script defines the priors for model parameters and initial values for a
    Wiener Diffusion Model. All priors are Gaussian, specified by their mean and variance (NOT
    standard deviation) in the space where they are estimated.

    The default values set here have a mean of 0 and a standard deviation of 4, though these can
    be customized for each parameter as required.

    The structure and methodologies are inspired from the HGF toolbox, open source code available as part of the TAPAS
    software collection: Frässle, S., et al. (2021). TAPAS: An Open-Source Software Package 
    for Translational Neuromodeling and Computational Psychiatry. Frontiers in Psychiatry, 12:680811. 
    https://www.translationalneuromodeling.org/tapas


    '''

    # Config structure as a dictionary
    c = {}
    
    # Model's name
    c['model'] = 'inverse gaussian: hgf'

    # Intercept of "a"
    c['a_amu'] = 0
    c['a_asa'] = 4

    # Slope of "a"
    c['b_amu'] = 0
    c['b_asa'] = 4

    # Drift V for valid response (resp == input)
    c['a_vmu'] = 0
    c['a_vsa'] = 4

    # Drift V for invalid response (resp != input)
    c['b_valmu'] = 0
    c['b_valsa'] = 4

    # Influence of muhat on "v"
    c['b_vmu'] = 0
    c['b_vsa'] = 4

    # Non-decision time
    c['Ter_mu'] = float('-inf')
    c['Ter_sa'] = 0

    ######## SET PRIORS #########
    c['priormus'] = np.concatenate([
        np.array([c['a_amu']]), 
        np.array([c['b_amu']]), 
        np.array([c['a_vmu']]), 
        np.array([c['b_valmu']]), 
        np.array([c['b_vmu']]), 
        np.array([c['Ter_mu']])
    ], axis=0)

    c['priorsas'] = np.concatenate([
        np.array([c['a_asa']]), 
        np.array([c['b_asa']]), 
        np.array([c['a_vsa']]), 
        np.array([c['b_valsa']]), 
        np.array([c['b_vsa']]), 
        np.array([c['Ter_sa']])
    ], axis=0)

    # Model file handle (function name as string)
    c['obs_fun'] = 'rdm_hgf'

    # Handle to function that transforms observation parameters to their native space
    c['transp_obs_fun'] = 'rdm_hgf_transp'

    return(c)

