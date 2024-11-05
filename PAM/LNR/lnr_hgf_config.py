import numpy as np 

def lnr_hgf_config():

    '''
    Contains the configuration for the Linear Deterministic Accumulator (LNR) according to:
    Heathcote, A., & Love, J. (2012). Linear Deterministic Accumulator Models of Simple Choice. Frontiers in Psychology, 3.

    The structure and methodologies are inspired from the HGF toolbox, open source code available as part of the TAPAS
    software collection: Frässle, S., et al. (2021). TAPAS: An Open-Source Software Package 
    for Translational Neuromodeling and Computational Psychiatry. Frontiers in Psychiatry, 12:680811. 
    https://www.translationalneuromodeling.org/tapas

    This configuration script defines the priors for model parameters and initial values for a
    LNR model. All priors are Gaussian, specified by their mean and variance (NOT
    standard deviation) in the space where they are estimated.
    
    The default values for all parameters, except non-decision time (Ter), 
    are pre-defined with the mean set to "0" and the standard deviation set to "4".

    The default values of the parameter "Ter" are set to "-Inf" for the mean
    and "0" for the standard deviation. With this combination of parameters
    the value of "Ter" is not estimated. It is possible to estimate "Ter"
    parameter by changing the values of the aforementioned parameters.
    It is possible to modify the specified values and assign custom values to each parameter.

    This config file does not take any input and returns the dict "c" as output. "c" contains all the necessary values to 
    define and build a LNR model.
    
    '''


    c = {}
    c['model'] = 'lnr: hgf'

    c['amu'] = 0
    c['asa'] = 4

    c['b_valmu'] = 0
    c['b_valsa'] = 4

    c['bmu'] = 0
    c['bsa'] = 4

    c['sigmamu'] = 0
    c['sigmasa'] = 4

    c['Termu'] = float('-inf')
    c['Tersa'] = 0
    
    ######## SET PRIORS #########
    c['priormus'] = np.concatenate([
        np.array([c['amu']]), 
        np.array([c['b_valmu']]), 
        np.array([c['bmu']]), 
        np.array([c['sigmamu']]), 
        np.array([c['Termu']])
    ], axis=0)
    
    c['priorsas'] = np.concatenate([
        np.array([c['asa']]), 
        np.array([c['b_valsa']]), 
        np.array([c['bsa']]), 
        np.array([c['sigmasa']]), 
        np.array([c['Tersa']])
    ], axis=0)

    c['obs_fun'] = 'lnr_hgf'          # model function name
    c['transp_obs_fun'] = 'lnr_hgf_transp'   # function name to transform obs. para > native space

    return(c)