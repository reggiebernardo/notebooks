import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis, moment
from scipy.stats import norm, normaltest, shapiro, monte_carlo_test
from tqdm import tqdm
from multiprocess import Pool, cpu_count

import healpy as hp
import json, os, glob, pickle, sys

from make_fpta_sims import *


def get_fft(t, r, Tspan_yr=30, n_bins=30, xx=10):
    '''compute Fourier components of a time series (t, r)'''
    mask = np.isfinite(r)
    ts=t-np.min(t) # corrects for phase in fft calculation
    interp_func = interp1d(ts[mask], r[mask], kind='cubic', fill_value="extrapolate")
    t_interp = np.linspace(0, (Tspan_yr*365.25), len(ts) * xx)
    r_interp = interp_func(t_interp)
    
    # fft calculation
    N = len(t_interp)
    a0_fft=np.sum(r_interp)*2/N
    a_fft=np.array([np.sum(r_interp*np.sin(2*np.pi*k*np.arange(N)/N)) for k in np.arange(1, n_bins+1)])*2/N
    b_fft=np.array([np.sum(r_interp*np.cos(2*np.pi*k*np.arange(N)/N)) for k in np.arange(1, n_bins+1)])*2/N

    # return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0=(1/Tspan_yr)*yrinv_to_nhz
    freqs=np.arange(1, n_bins+1)*f0
    return a0_fft, a_fft, b_fft, freqs


def bin_res_in_k(psrs, n_bins=30, tmin_mjd=0):
    '''bin pulsar residuals in fourier space'''
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan=np.max(tmax)-np.min(tmin)
    Tspan_yr=np.round(Tspan/86400/365.25,1)

    a0_bins = []
    a_bins = []
    b_bins = []
    freqs_bins = []
    pos_bins = []

    for psr in psrs:
        a0_psr, a_psr, b_psr, freqs_psr=\
        get_fft(tmin_mjd+psr.toas/86400, psr.residuals, Tspan_yr=Tspan_yr, n_bins=n_bins)
        a0_bins.append(a0_psr)
        a_bins.append(a_psr)
        b_bins.append(b_psr)
        freqs_bins.append(freqs_psr)
        pos_bins.append(psr.pos)

    a0_bins=np.array(a0_bins)
    a_bins=np.array(a_bins)
    b_bins=np.array(b_bins)
    freqs_bins=np.array(freqs_bins)
    pos_bins=np.array(pos_bins)
    
    return a0_bins, a_bins, b_bins, freqs_bins, pos_bins

def get_ops(data):
    '''compute one-point statistics of data'''
    return np.array([np.mean(data), np.var(data), \
                     moment(data,moment=3), moment(data,moment=4)])
    # return np.array([np.mean(data), np.var(data), \
    #                  skew(data,bias=False), kurtosis(data,bias=False)])
    # mean_data=np.mean(data)
    # delta_data=(data-mean_data)/mean_data
    # return np.array([np.mean(delta_data), np.var(delta_data), \
    #                  moment(delta_data,moment=3), moment(delta_data,moment=4)])

def get_op_stats_k(a0_bins, a_bins, b_bins, freqs_bins):
    '''compute one-point statistics given pta fourier data'''
    a0_stats = get_ops(a0_bins)
    # get a, b stats
    freqs=np.mean(freqs_bins, axis=0)
    a_stats = np.array([get_ops(a_bins[:, i]) for i in range(len(freqs))])
    b_stats = np.array([get_ops(b_bins[:, i]) for i in range(len(freqs))])
    return a0_stats, a_stats, b_stats, freqs

# for mc resampled p-value check of normality
def mctest_statistic(x, axis):
    return normaltest(x, axis=axis).statistic

def get_pvals(data):
    '''Gaussianity standard tests'''
    return np.array([shapiro(data).pvalue, normaltest(data).pvalue, \
                     monte_carlo_test(data, norm.rvs, mctest_statistic, alternative='greater').pvalue])

def get_pvals_k(a0_bins, a_bins, b_bins, freqs_bins):
    a0_pvals = get_pvals(a0_bins)
    # get a, b stats
    freqs=np.mean(freqs_bins, axis=0)
    a_pvals = np.array([get_pvals(a_bins[:, i]) for i in range(len(freqs))])
    b_pvals = np.array([get_pvals(b_bins[:, i]) for i in range(len(freqs))])
    return a0_pvals, a_pvals, b_pvals, freqs


# time binning functions

def bin_res_in_time(psrs, n_bins=30):
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    
    bin_edges = np.linspace(np.min(tmin), np.max(tmax), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bins_residuals = [[] for _ in range(n_bins)]
    bins_pos = [[] for _ in range(n_bins)]
    bins_toas = [[] for _ in range(n_bins)]
    
    total_iterations = sum(len(psr.toas) for psr in psrs)
    
    with tqdm(total=total_iterations, desc="Progress", unit=" iterations") as pbar:
        # iterate over each pulsar in psrs
        for psr in psrs:
            bin_indices = np.digitize(psr.toas, bin_edges) - 1
            
            # assign residuals and toas to their corresponding bins
            for i, bin_index in enumerate(bin_indices):
                if 0 <= bin_index < n_bins:  # if bin_index is within valid range
                    bins_toas[bin_index].append(psr.toas[i])
                    bins_residuals[bin_index].append(psr.residuals[i])
                    bins_pos[bin_index].append(psr.pos)
                
                pbar.update(1)  # update progress bar for each iteration
    
    return bin_edges, bins_toas, bins_residuals, bins_pos


def get_op_stats(bins_toas, bins_residuals):
    pta_mean = []
    pta_vars = []
    pta_skew = []
    pta_kurt = []

    for i in range(len(bins_toas)):
        res = np.array( bins_residuals[i] )
        pta_mean.append( np.mean(res) )
        pta_vars.append( np.var(res) )
        pta_skew.append( skew(res, bias=False) )
        pta_kurt.append( kurtosis(res, bias=False) )
    
    pta_mean = np.array(pta_mean)
    pta_vars = np.array(pta_vars)
    pta_skew = np.array(pta_skew)
    pta_kurt = np.array(pta_kurt)
    return pta_mean, pta_vars, pta_skew, pta_kurt


# master function for one-point statistics
def opstats_master(npsrs=50, Tspan_yr=30, nsims=30, add_gwb=False, gamma_gw=13/3, log10_A_gw=np.log10(2.4e-15), \
                   add_red=False, add_white=False, n_bins=14, ntoas=100, toaerr=1e-6, save_pickle=True, \
                   plot_stats=False, plot_st_index=3):
    
    # Redirect stdout and stderr to suppress output
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        # code starts here
        pta_sims = []
        for i in range(nsims):
            psrs = make_fpta_sims(npsrs=npsrs, Tobs=Tspan_yr, ntoas=ntoas, toaerr=toaerr, \
                                  add_gwb=add_gwb, gamma_gw=gamma_gw, log10_A_gw=log10_A_gw, \
                                  add_red=add_red, add_white=add_white, save_pickle=False)
            pta_sims.append(psrs)

    finally:
        # Reset stdout and stderr to their original values
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # compute ak and bk given residuals data
    pta_sims_fbins = {}
    for i, psrs in enumerate(pta_sims):  # loop over mock data sets
        a0_bins, a_bins, b_bins, freqs_bins, _ = bin_res_in_k(psrs, n_bins=n_bins)
        pta_sims_fbins[f'mock_{i}'] = (a0_bins, a_bins, b_bins, freqs_bins)

    # compute sample statistics of ak and bk
    pta_sims_stats = {}
    for i in range(len(pta_sims)):  # loop over mock data sets
        a0_i, a_i, b_i, f_i = pta_sims_fbins[f'mock_{i}']
        a0_stats_i, a_stats_i, b_stats_i, freqs_i = get_op_stats_k(a0_i, a_i, b_i, f_i)
        pta_sims_stats[f'mock_{i}'] = (a0_stats_i, a_stats_i, b_stats_i, freqs_i)

    # simplifying sample statistics
    a0_stats_compiled = []
    a_stats_compiled = []
    b_stats_compiled = []
    freqs_compiled = []
    for i in range(len(pta_sims_stats)):
        a0_stats_i, a_stats_i, b_stats_i, freqs_i = pta_sims_stats[f'mock_{i}']
        a0_stats_compiled.append(a0_stats_i)
        a_stats_compiled.append(a_stats_i)
        b_stats_compiled.append(b_stats_i)
        freqs_compiled.append(freqs_i)
    a0_stats_compiled = np.array(a0_stats_compiled)
    a_stats_compiled = np.array(a_stats_compiled)
    b_stats_compiled = np.array(b_stats_compiled)
    freqs_compiled = np.array(freqs_compiled)

    freqs_bins = np.mean(freqs_compiled, axis=0)

    # main results in a dictionary
    pta_ensemble_pvals = {}
    pta_ensemble_pvals['rk_stats'] = {'a0': a0_stats_compiled, \
                                        'ak': a_stats_compiled, 'bk': b_stats_compiled}
    pta_ensemble_pvals['freqs_bins'] = freqs_bins

    # st_index=2 # st_index=0 (mean), 1 (variance), 2 (skewness), 3 (kurtosis)
    keys_stats = ['meanpvals', 'varipvals', 'skewpvals', 'kurtpvals']
    for st_index in range(len(keys_stats)):
        stats_dict = {}
        # loop ensemble over frequencies index i
        stats_dict['ak'] = np.array([get_pvals(a_stats_compiled[:, i, st_index]) for i in range(len(freqs_bins))])
        stats_dict['bk'] = np.array([get_pvals(b_stats_compiled[:, i, st_index]) for i in range(len(freqs_bins))])

        pta_ensemble_pvals[keys_stats[st_index]] = stats_dict

    pta_ensemble_pvals['input'] = {'npsrs': npsrs, 'Tspan_yr': Tspan_yr, 'nsims': nsims, \
                                   'add_gwb': add_gwb, 'gamma_gw': gamma_gw, 'log10_A_gw': log10_A_gw, 'add_red': add_red, \
                                   'add_white': add_white, 'n_bins': n_bins, 'ntoas': ntoas, 'toaerr': toaerr}

    if add_gwb:
        gwb_binary = 1
    else:
        gwb_binary = 0

    if add_red:
        rn_binary = 1
    else:
        rn_binary = 0

    if save_pickle:
        # Define the directory path
        dir_path = './data'

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the pickle file
        pkl_file = 'opstats' + f'_npsrs{npsrs}_nsims{nsims}_gwb{gwb_binary}_rn{rn_binary}'
        with open(f'{dir_path}/{pkl_file}.pkl', 'wb') as f:
            pickle.dump(pta_ensemble_pvals, f)

    if plot_stats:
        # plot p-values (null hypothesis=Gaussian distribution)
        key_stats = keys_stats[plot_st_index]

        fig, ax = plt.subplots(nrows=2)
        ax[0].errorbar(freqs_bins, pta_ensemble_pvals[key_stats]['ak'][:, 0], fmt='ro', alpha=0.6)
        ax[0].errorbar(freqs_bins, pta_ensemble_pvals[key_stats]['ak'][:, 1], fmt='bs', alpha=0.6)
        ax[0].errorbar(freqs_bins, pta_ensemble_pvals[key_stats]['ak'][:, 2], fmt='g^', alpha=0.6)

        ax[1].errorbar(freqs_bins, pta_ensemble_pvals[key_stats]['bk'][:, 0], fmt='ro', alpha=0.6, label='Shapiro-Wilk')
        ax[1].errorbar(freqs_bins, pta_ensemble_pvals[key_stats]['bk'][:, 1], fmt='bs', alpha=0.6, label="D'Agostino-Pearson")
        ax[1].errorbar(freqs_bins, pta_ensemble_pvals[key_stats]['bk'][:, 2], fmt='g^', alpha=0.6, label='DP-MC')

        ax[0].axhline(y=0.05, ls=':', color='magenta')
        ax[1].axhline(y=0.05, ls=':', color='magenta')

        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[1].legend(loc='lower right')
        ax[0].set_ylabel(rf'$p$-value ($\overline{{a_k^{{{plot_st_index+1}}}}}$)')
        ax[1].set_ylabel(rf'$p$-value ($\overline{{b_k^{{{plot_st_index+1}}}}}$)')
        ax[len(ax)-1].set_xlabel('Frequency (nHz)')
        ax[0].set_title(rf'GWB={add_gwb}, RN={add_red}, $N_p$={npsrs}')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.00)
        plt.show()

    return pta_ensemble_pvals


# loading and plotting utility functions

def load_ensemble_stats(npsrs, nsims, gwb_binary, rn_binary, print_input=False):
    dir_path = './data'
    pkl_file='opstats' + f'_npsrs{npsrs}_nsims{nsims}_gwb{gwb_binary}_rn{rn_binary}'
    
    pickle_loc = os.path.join(dir_path, pkl_file + '.pkl')
    if os.path.exists(pickle_loc):
        with open(pickle_loc, 'rb') as f:
            pta_ensemble_pvals = pickle.load(f)
        if print_input:
            print(pta_ensemble_pvals['input'])
        return pta_ensemble_pvals
    else:
        print('pre-computed stats do not exist')

def plot_violin(ax, x, y_data, widths=0.5, color='blue', quantiles=[0.16,0.84], \
                show_medians=True, show_extrema=False, alpha=0.3, zorder=0, use_logx=True):
    if use_logx:
        x= np.log10(x*1e-9)
    vp = ax.violinplot(y_data, positions=[x], widths=widths, \
                       quantiles=quantiles, showmedians=show_medians, showextrema=show_extrema)
    vp['cmedians'].set_colors(color)
    vp['cquantiles'].set_colors(color)
    vp['bodies'][0].set_zorder(zorder)
    for pc in vp['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(alpha)
        pc.set_linestyle('None')

def plot_rk_moments(ax, opstats, widths=0.1, color='blue', quantiles=[0.16,0.84], \
                    show_medians=True, show_extrema=False, alpha=0.3, zorder=0, use_logx=True):
    '''ax is 4x2, opstats is output from xxx_opstats_master'''
    freqs_bins=opstats['freqs_bins']
    rk_stats=opstats['rk_stats']

    def plot_violin_short(axij, x, y_data):
        plot_violin(axij, x, y_data, \
                    widths=widths, color=color, quantiles=quantiles, \
                    show_medians=show_medians, show_extrema=show_extrema, \
                    alpha=alpha, zorder=zorder, use_logx=use_logx)
    
    for i in range(rk_stats['ak'].shape[1]):
        # mean and variance
        plot_violin_short(ax[0,0], freqs_bins[i], rk_stats['ak'][:,i,0]/1e-6)
        plot_violin_short(ax[1,0], freqs_bins[i], rk_stats['ak'][:,i,1]/1e-12)
        plot_violin_short(ax[0,1], freqs_bins[i], rk_stats['bk'][:,i,0]/1e-6)
        plot_violin_short(ax[1,1], freqs_bins[i], rk_stats['bk'][:,i,1]/1e-12)
        
        # skewness and kurtosis
        plot_violin_short(ax[2,0], freqs_bins[i], rk_stats['ak'][:,i,2]/1e-18)
        plot_violin_short(ax[3,0], freqs_bins[i], rk_stats['ak'][:,i,3]/1e-24)
        plot_violin_short(ax[2,1], freqs_bins[i], rk_stats['bk'][:,i,2]/1e-18)
        plot_violin_short(ax[3,1], freqs_bins[i], rk_stats['bk'][:,i,3]/1e-24)

def plot_normality_pvals(ax, opstats, st_index=0, pval_type_index=0, fmt='ro', alpha=0.8, label=None):
    '''ax (nrows=2) each for ak and bk'''
    log10_A_gw=opstats['input']['log10_A_gw']
    if log10_A_gw < -18:
        add_gwb=False
    else:
        add_gwb=True
    add_red=opstats['input']['add_red']
    
    # st_index=0 (mean), 1 (variance), 2 (skewness), 3 (kurtosis)
    keys_stats=['meanpvals', 'varipvals', 'skewpvals', 'kurtpvals']
    key_stats=keys_stats[st_index]
    
    # plot p-values (null hypothesis=Gaussian distribution)
    # p-values types: pval_type_index=0 (Shapiro-Wilk), 1 (DAgostino-Pearson), 2 (DP-MC)
    freqs_bins=opstats['freqs_bins']
    ax[0].errorbar(freqs_bins, opstats[key_stats]['ak'][:, pval_type_index], \
                   fmt=fmt, alpha=alpha, label=label)    
    ax[1].errorbar(freqs_bins, opstats[key_stats]['bk'][:, pval_type_index], fmt=fmt, alpha=alpha)


def plot_rk_moments_overlay(ax, opstats, widths=0.1, color='blue', quantiles=[0.16,0.84], \
                            show_medians=True, show_extrema=False, alpha=0.3, zorder=0, use_logx=True):
    '''ax is 4x1, opstats is output from xxx_opstats_master'''
    freqs_bins=opstats['freqs_bins']
    rk_stats=opstats['rk_stats']

    def plot_violin_short(axij, x, y_data):
        plot_violin(axij, x, y_data, \
                    widths=widths, color=color, quantiles=quantiles, \
                    show_medians=show_medians, show_extrema=show_extrema, \
                    alpha=alpha, zorder=zorder, use_logx=use_logx)
    
    for i in range(rk_stats['ak'].shape[1]):
        # mean and variance
        plot_violin_short(ax[0], freqs_bins[i], rk_stats['ak'][:,i,0]/1e-6)
        plot_violin_short(ax[1], freqs_bins[i], rk_stats['ak'][:,i,1]/1e-12)
        plot_violin_short(ax[0], freqs_bins[i], rk_stats['bk'][:,i,0]/1e-6)
        plot_violin_short(ax[1], freqs_bins[i], rk_stats['bk'][:,i,1]/1e-12)
        
        # skewness and kurtosis
        plot_violin_short(ax[2], freqs_bins[i], rk_stats['ak'][:,i,2]/1e-18)
        plot_violin_short(ax[3], freqs_bins[i], rk_stats['ak'][:,i,3]/1e-24)
        plot_violin_short(ax[2], freqs_bins[i], rk_stats['bk'][:,i,2]/1e-18)
        plot_violin_short(ax[3], freqs_bins[i], rk_stats['bk'][:,i,3]/1e-24)

# if __name__ == "__main__":
#     opstats_master()
