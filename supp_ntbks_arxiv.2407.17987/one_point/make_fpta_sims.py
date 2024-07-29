from fakepta import fake_pta
from fakepta.fake_pta import Pulsar, make_fake_array
from fakepta.correlated_noises import add_common_correlated_noise

import numpy as np
import matplotlib.pyplot as plt
from healpy.newvisufunc import newprojplot

import os
import pickle


def make_fpta_sims(npsrs=100, Tobs=30, ntoas=100, toaerr=1e-7, \
                   add_gwb=False, gamma_gw=13/3, log10_A_gw=-15, \
                   add_red=False, add_white=False, save_pickle=False, pkl_file='my_pickle'):
        
    psrs = make_fake_array(npsrs=npsrs, Tobs=Tobs, ntoas=ntoas, \
                           isotropic=False, gaps=True, toaerr=toaerr, \
                           pdist=1., backends='NUPPI.1400', noisedict=None, \
                           custom_model={'RN':30, 'DM':100, 'Sv':None})
                
    for psr in psrs:
        psr.make_ideal()  # Flattens out the residuals
        if add_red:
            psr.add_red_noise(spectrum='powerlaw', \
                              log10_A=np.random.uniform(-16, -13), \
                              gamma=np.random.uniform(2, 6))
        if add_white:
            psr.add_white_noise(add_ecorr=True)

        # fakepta white noise parameters, for reference 
        # https://github.com/mfalxa/fakepta/blob/main/fakepta/fake_pta.py
            # for key in [*self.noisedict]:
            #     if 'efac' in key:
            #         self.noisedict[key] = np.random.uniform(0.5, 2.5)
            #     if 'equad' in key:
            #         self.noisedict[key] = np.random.uniform(-8., -5.)
            #     if add_ecorr and 'ecorr' in key:
            #         self.noisedict[key] = np.random.uniform(-10., -7.)
    
    if add_gwb:
        add_common_correlated_noise(psrs,  spectrum='powerlaw', orf='hd', \
                                    log10_A=log10_A_gw, gamma=gamma_gw)

    if save_pickle:
        # Define the directory path
        dir_path = './fpta_sims'
        
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Save the pickle file
        with open(f'{dir_path}/{pkl_file}.pkl', 'wb') as f:
            pickle.dump(psrs, f)

    print(f'Created FPTA {Tobs}-yr data with {npsrs} pulsars')
    return psrs


def load_fpta_sims(pkl_file):
    dir_path = './fpta_sims'
    psrs = pickle.load(open(f'{dir_path}/{pkl_file}.pkl', 'rb'))

    # get number of pulsars
    npsrs=len(psrs)

    # find the maximum time span
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    Tspan_mjd = Tspan/86400
    Tspan_yr = Tspan_mjd/365.25

    print(f'Loaded FPTA {int(Tspan_yr)}-yr data with {npsrs} pulsars')
    return psrs, npsrs, Tspan_yr
    

def splot_pulsars(psrs, color='r', marker='*', markersize=10, \
                  alpha=0.5, if_projplot=False):
    
    if if_projplot: # using healpy newprojplot
        for psr in psrs:
            newprojplot(theta=psr.theta, phi=psr.phi-np.pi, \
                        fmt=marker, color=color, markersize=markersize, alpha=alpha)
    else:
        for psr in psrs:
            plt.scatter(np.pi - np.array(psr.phi), np.pi/2 - np.array(psr.theta), \
                        marker=marker, color=color, s=markersize**2, alpha=alpha)

