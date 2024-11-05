import numpy as np 

def lnr_hgf_config():
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